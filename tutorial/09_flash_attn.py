'''
Docstring for tutorial.flash_attn
- call sub-kernels
- using tl.exp2()
- specific flash-attention
- tl.static_assert()
- multi-axis launch grid
- pre-compute in seperate kernel
- using approximate constants

DOES NOT INCLUDE
- only fp32
- dropout
'''


import torch
import triton
import triton.language as tl

import math

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


@triton.jit
def _attn_fwd_inner(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    block_index_QO,
    softmax_scale,
    stride_K_N, stride_V_N,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
    N: tl.constexpr, Dh: tl.constexpr,
):
    """
    arrows indicate direction of this pid's for loop; each arrow is a different PID
                N of K & V
                ------------>
                ------------>
    N of Q      ------------>
                ------------>
                ------------>
    but if we actually take into account the causal mask then really it's more like
                N of K & V
                >
                --->
    N of Q      ------>
                --------->
                ------------>
    and to get even more accurate, we do the diagonal in our second call of this inner kernel
                N of K & V
                x
                   x
    N of Q            x
                         x
                            x
    and then the first call gets all the parts below the diagonal
                N of K & V
                
                -->
    N of Q      ----->
                -------->
                ----------->
    """
   
    if DIAGONAL:
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
        lo = tl.multiple_of(lo, BLOCK_SIZE_QO)
        '''
        and to get even more accurate, we do the diagonal in our second call of this inner kernel
                    N of K & V
                    x
                       x
        N of Q            x
                             x
                                x
        '''
    else:
       lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_KV is a multiple of BLOCK_SIZE_KV, so the compiler can do optimizations
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)

        mask_KV_N = offsets_KV_N < N

        K_T = tl.load(K_ptr+ K_T_offsets, mask=mask_KV_N[None, : ], other=0.) # shape (Dh, BLOCK_SIZE_KV)

        S = tl.dot(Q, K_T) * softmax_scale # shape ( BLOCK_SIZE_QO, BLOCK_SIZE_KV)

        if DIAGONAL:
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            S = tl.where(causal_mask, S, -1.0e6) # Sjape (BLOCK_SIZE_QO , BLOCKS_SIZEKV)

        M_new = tl.maximum(M, tl.max(S, axis=1))

        S -= M_new[:, None]
        P = tl.exp2(S)
        L_new = tl.sum(P, axis=-1)
        alpha = tl.exp2(M - M_new)  # correction factor

        L = L * alpha + L_new

        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:,None], other=0.0)

        O = O * alpha[:, None]
        O = tl.dot(P,V, acc=O)

        M = M_new
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O, L, M


@triton.autotune(
    [
        triton.Config({"PRE_BLOCK_SIZE_ROW": PRE_BLOCK_SIZE_ROW},
                        num_stages=num_stages, num_warps=num_warps,)
        for PRE_BLOCK_SIZE_ROW in [32]#, 64, 128, 256]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
    ],
    key=["Dh"],
)
@triton.jit
def attn_backward_preprocess(
    O_ptr, dLdO_ptr, Delta_ptr,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_dLdO_B, stride_dLdO_H, stride_dLdO_N, stride_dLdO_Dh,
    stride_Delta_B,stride_Delta_H,stride_Delta_N,
    N, Dh: tl.constexpr,
    PRE_BLOCK_SIZE_ROW: tl.constexpr
):
    """the job of this kernel is to pre-compute Delta since Delta is used by both of the following two kernels"""
    index_BH = tl.program_id(1)
    row = tl.program_id(0)


    row_offset = row * PRE_BLOCK_SIZE_ROW + tl.arange(0, PRE_BLOCK_SIZE_ROW)
    col_offset = tl.arange(0, Dh)
    mask = row_offset < N

    O_ptr += index_BH * stride_O_H
    O_offsets = row_offset[:, None] * stride_O_N + col_offset[None, :] * stride_O_Dh

    O_val = tl.load(O_ptr + O_offsets, mask=mask[:,None], other=0.0)

    dLdO_ptr += index_BH * stride_dLdO_H
    dLdO_offsets = row_offset[:, None] * stride_dLdO_N + col_offset[None, :] * stride_dLdO_Dh
    dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask = mask[:, None], other=0.)


    Delta = tl.sum(dLdO.to(tl.float32) * O_val.to(tl.float32), axis=1) # shape (PRE_BLOCK_SIZE_ROW)
    Delta_ptr += index_BH * stride_Delta_H
    tl.store(Delta_ptr + row_offset, Delta, mask=mask)


@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_QO in [16]#, 32, 64, 128]
        for BLOCK_SIZE_KV in [16]#, 32, 64, 128]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
    ],
    key=["Dh"],
)
@triton.jit
def attn_fwd(
    q_ptr, k_ptr ,v_ptr,o_ptr, LSE_ptr,
    softmax_scale,
    q_B_stride,q_H_stride,q_N_stride,q_Dh_stride,
    k_B_stride,k_H_stride,k_N_stride,k_Dh_stride,
    v_B_stride,v_H_stride,v_N_stride,v_Dh_stride,
    O_B_stride,O_H_stride,O_N_stride,O_Dh_stride,
    LSE_B_stride,LSE_H_stride,LSE_N_stride,
    B, 
    H: tl.constexpr, N: tl.constexpr, Dh: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr,BLOCK_SIZE_KV: tl.constexpr, 
):
    # scale softmax by ln2 if using tl.exp2
    rln2: tl.constexpr = 1.4426950408889634 # 1/ math.log(2)
    softmax_scale *= rln2

    """
    let's show that e^x = 2^(x * rln2)
    e^x = (2^(log_2(e)))^x since a = 2^log_2(a)
    then using the power rule 
    (2^(log_2(e)))^x = 2^(x * log_2(e))
    log_2(e) = 1/log_e(2)
    e^x = 2^(x* 1/log_e(2))
    """
    # not sure 
    tl.static_assert(BLOCK_SIZE_KV <= Dh)
    tl.static_assert(BLOCK_SIZE_QO <= N)
    tl.static_assert(BLOCK_SIZE_KV <= N)



    block_index_QO = tl.program_id(0)
    index_BH = tl.program_id(1)

    index_B = index_BH // H
    index_H = index_BH % H



    q_ptr += index_B * q_B_stride + index_H * q_H_stride
    k_ptr += index_B * k_B_stride + index_H * k_H_stride
    v_ptr += index_B * v_B_stride + index_H * v_H_stride
    o_ptr += index_B * O_B_stride + index_H * O_H_stride

    offset_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0,BLOCK_SIZE_QO)
    offset_KV_N = tl.arange(0,BLOCK_SIZE_KV)
    offset_Dh = tl.arange(0, Dh)

    Q_offsets = offset_QO_N[:, None] * q_N_stride + offset_Dh[None, :] * q_Dh_stride
    K_T_offsets = offset_KV_N[None, :] * k_N_stride + offset_Dh[:, None] * k_Dh_stride
    V_offsets = offset_KV_N[:, None] * v_N_stride + offset_Dh[None, :] * v_Dh_stride


    # varies 
    """
    arrows indicate direction of this pid's for loop; each arrow is a different PID
                N of K & V
                ------------>
                ------------>
    N of Q      ------------>
                ------------>
                ------------>
    """
    mask_QO_N = offset_QO_N < N
    Q = tl.load(q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.0) # shape (BLOCK_SIZE_QO, Dh)


    # allocate tensors for stroing intermediate & output values 
    # the running maximum. We have one entry for each query in the block we're currently working on 
    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32)
    # the running sum. We have one entry for each query (since we sum the attention scores by rows)
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32)
    # the accumulator for the output, which is a group of rows of the O matrix
    O = tl.zeros(shape=[BLOCK_SIZE_QO, Dh], dtype=tl.float32)
    

    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        k_ptr, v_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        k_N_stride, v_N_stride,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False, # blocks on the DIAGONAL get special treatment if this is set to true; we use it below
        offset_QO_N, offset_KV_N,
        N, Dh,
    )

     # This step runs for the blocks on the diagonal in the causal attention mask
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        k_ptr, v_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        k_N_stride, v_N_stride,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True, # blocks on the diagonal get special masking treatment
        offset_QO_N, offset_KV_N,
        N, Dh,
    )

    O = O / L[:, None]

    LSE = M + tl.math.log2(L) # shape (BLOCK_SIZE_QO)

    """
    softmax(x_i) = exp(x_i - m_i)/ l_i
    = exp(x_i - m_i)/ exp(log(l_i))
    = exp(x_i - (m_i + log(l_i))
    """

    LSE_offsets = index_BH * LSE_H_stride + offset_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N 
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)

    O_offsets = offset_QO_N[:, None] * O_N_stride + offset_Dh[None, :] * O_Dh_stride
    tl.store(o_ptr + O_offsets, O ,mask=mask_QO_N[:, None]) # shape (BLOCKS_SIZE_Q, Dh)


@triton.jit
def _attn_backward_KV(
    K, V, dLdK, dLdV,               # shape (BLOCK_SIZE_COL, D)
    Q_ptr, dLdO_ptr,
    LSE_ptr, Delta_ptr, 
    stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,   # no more _1 because this sub-kernel is the _1
    BLOCK_SIZE_COL: tl.constexpr, 
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    """
    this sub-kernel will be looking at a specific chunk of K & V , where we call
    the sequence length of K & V the columns of our NxN attention matrix,
    and iterating through rows of Q's sequence length to calculate that 
    chunk of dLdK and dLdV
                    N of K & V
               |    |   |   |   |
               |    |   |   |   |
    N of Q     |    |   |   |   |
               |    |   |   |   |
              \|/  \|/ \|/ \|/ \|/
    arrows indicate direction of this pid's for loop; each arrow is a different PID
    """
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    # we transpose Q while loading it rather than in a separate kernel
    Q_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_ROW[None, :] * stride_N
    dLdO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh

    for block_idx in range(num_steps):
        # we load M before computing S to reduce pipeline stall (and dLdO before computing dLdV)
        # meaning the Triton compiler can have an easier time doing the loading of M
        # and the dot product of K and QT simultaneously. in general you should load a bunch
        # of stuff then calc a bunch of stuff rather than flipping b/w loads and calcs
        mask_N = offsets_ROW < N
        Q_T = tl.load(Q_ptr + Q_T_offsets, mask=mask_N[None, :], other=0.) # shape (Dh, BLOCK_SIZE_ROW)
        LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_N, other=0.) # shape (BLOCK_SIZE_ROW)
        dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask_N[:, None], other=0.) # shape (BLOCK_SIZE_ROW, Dh)
        Delta = tl.load(Delta_ptr + offsets_ROW, mask=mask_N, other=0.) # shape (BLOCK_SIZE_ROW)
        # ^notice the order we load these in is based on the order we use them below

        # we'll re-calculate transpose of S and P matrices since doing that here is faster & more importantly
        #  cheaper on memory consumption than if we were to have saved them in our forward pass & read them here
        S_T = tl.dot(K, Q_T) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
            # no scale here because the operation is associative so we did it earlier on K
            # thanks to masking of K & Q_T, the non-existent out-of-bounds tokens look like a bunch
            #  of zeros hugged up against the bottom and right edges of S_T
        # subtract S_T by the logsumexp then exponentiate to get P_T
        P_T = tl.exp2(S_T - LSE[None, :]) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
            # this derivative actually requires an extra *ln(2) which we do below at dLdS_T
            # the non-existent tokens that were a bunch of 0's are now a bunch of 1's

        if MASK: # if we're on the block-diagonal
            # implement a lower-triangular mask. it looks like upper-triangular because we've 
            #  transposed, which is also the reason why our columns & rows are reversed
            mask = (offsets_COL[:, None] <= offsets_ROW[None, :]) # (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
            P_T = tl.where(mask, P_T, 0.)

        # compute dLdV
        dLdV = tl.dot(P_T, dLdO, acc=dLdV) # shape (BLOCK_SIZE_COL, Dh)

        # compute dLdP_T and dLdS_T to get dLdK
        dLdP_T = tl.dot(V, tl.trans(dLdO)) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        dLdS_T = (P_T * (dLdP_T - Delta[None, :]) * ln2) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        dLdK = tl.dot(dLdS_T, tl.trans(Q_T), acc=dLdK) # shape (BLOCK_SIZE_COL, D)
            # acc tells the tl.dot to accumulate into dLdK

        # increment pointers
        offsets_ROW += BLOCK_SIZE_ROW
        Q_ptr += BLOCK_SIZE_ROW * stride_N
        dLdO_ptr += BLOCK_SIZE_ROW * stride_N
    
    return dLdK, dLdV

@triton.jit
def _attn_backward_Q(
    dLdQ, Q, dLdO, LSE, 
    K_ptr, V_ptr, Delta_ptr,
    stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr, 
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    """
    this sub-kernel will be looking at a specific chunk of Q and iterating through
    rows of K & V to calculate that chunk of dLdQ
    I say "rows" of K and V but really we refer to them as colums since we're thinking
    not in terms of the (B, H, N, D) shaped matrices but rather the (B, H, N, N) shaped
    attention logits, where the first N are split up by "BLOCK_SIZE_ROW" and the second N
    is split up by "BLOCK_SIZE_COL"
                    N of K & V
               ------------------->
               ------------------->
    N of Q     ------------------->
               ------------------->
               ------------------->
    arrows indicate direction of this pid's for loop; each arrow is a different PID
    """
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    # we transpose V while loading it
    K_and_V_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N

    Delta = tl.load(Delta_ptr + offsets_ROW, mask=offsets_ROW<N, other=0.) # shape (BLOCK_SIE_ROW)

    for block_idx in range(num_steps):
        K_T = tl.load(K_ptr + K_and_V_T_offsets, mask=(offsets_COL < N)[None, :], other=0.) 
            # shape (Dh, BLOCK_SIZE_COL)
        V_T = tl.load(V_ptr + K_and_V_T_offsets, mask=(offsets_COL < N)[None, :], other=0.) 
            # shape (Dh, BLOCK_SIZE_COL)

        S = tl.dot(Q, K_T) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
            # no scale here because the operation is associative so we did it earlier on Q
        P = tl.exp2(S - LSE) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)

        if MASK: # if we're on the block-diagonal
            mask = (offsets_ROW[:, None] >= offsets_COL[None, :]) # (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
            # setting lower-triangular values to zero since the gradient is upper-triangular
            P = tl.where(mask, P, 0.) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)

        # calc dLdP and dLdS to get dLdQ
        dLdP = tl.dot(dLdO, V_T) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        dLdS = (P * (dLdP - Delta[:, None]) * ln2) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
            # ^this line is equivalent to:
            #weighted_dLdP = tl.sum(dLdP * P, axis=1)  # row-sum over keys
            #dLdS = P * (dLdP - weighted_dLdP[:, None])
            # but trades-off a memory access for a binary & then a reduction op
        dLdQ += tl.dot(dLdS, tl.trans(K_T)) # shape (BLOCK_SIZE_ROW, Dh)
            # we'll need to de-sdcale dLdQ in the end because K_T was pre-scaled
            # we do it later instead of now bc now would mean num_steps * flops versus just flops
        
        # increment pointers
        offsets_COL += BLOCK_SIZE_COL
        K_ptr += BLOCK_SIZE_COL * stride_N
        V_ptr += BLOCK_SIZE_COL * stride_N
    
    return dLdQ



@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MACRO": BLOCK_SIZE_MACRO, "BLOCK_SIZE_MICRO": BLOCK_SIZE_MICRO},
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_MICRO in [16]#, 32, 64]
        for BLOCK_SIZE_MACRO in [32]#, 64, 128]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
        if BLOCK_SIZE_MACRO > BLOCK_SIZE_MICRO # could do >= but i wanna get mileage out of the loop code we wrote
    ],
    key=["Dh"],
)
@triton.jit
def attn_backward(
    Q_ptr, K_ptr, V_ptr, 
    dLdO_ptr, dLdQ_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, Delta_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr, 
    BLOCK_SIZE_MICRO: tl.constexpr,  #
    BLOCK_SIZE_MACRO: tl.constexpr,  #
):
    # we'll use these constants later on Q
    ln2: tl.constexpr = 0.6931471824645996  # = ln(2), natural logarithm of 2
    rln2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2), the reciprocal of the natural logarithm of 2
        # generally defining a known constant as an approximation of itself to some number of digits
        #  is more efficient than calculating the actual value every time

    # move pointers of (B, H, N, D) matrices to get to the correct batch and head
    idx_batch_head = tl.program_id(1)
    idx_batch = idx_batch_head // H
    idx_head = idx_batch_head % H 
    batch_head_jump = idx_batch * stride_B + idx_head * stride_H
    Q_ptr += batch_head_jump
    K_ptr += batch_head_jump
    V_ptr += batch_head_jump
    dLdO_ptr += batch_head_jump
    dLdQ_ptr += batch_head_jump
    dLdK_ptr += batch_head_jump
    dLdV_ptr += batch_head_jump

    # move pointers of (B, H, N) matrices to get to the correct batch and head
    batch_head_jump = idx_batch_head * N
    LSE_ptr += batch_head_jump
    Delta_ptr += batch_head_jump

    # BLOCK_SIZE_MACRO must be a multiple of BLOCK_SIZE_MICRO
    # because we use them to determine num_steps and don't want a remainder
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    ### STAGE 1: First we'll do dLdK and dLdV
    # in the fwd loop we held a block of Q in SRAM and iterated through K & V; here we'll do the opposite

    # ROW and COL refer to the N dimension of (Q & O) and (K & V) respectively
    # for stage 1 each PID will look at BLOCK_SIZE_MACRO tokens of K & V and there will 
    #  be an inner for-loop iterating over BLOCK_SIZE_MICRO tokens of Q at a time
    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO

    # first we'l do the gradients along the block diagonal since they get treated differently
    #  in that they have an triangular causal mask
    pid = tl.program_id(0)
    start_COL = pid * BLOCK_SIZE_COL_1
    start_ROW = start_COL
    num_steps = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1
    
    # load K & V
    offsets_COL_1 = start_COL + tl.arange(0, BLOCK_SIZE_COL_1)
    offsets_Dh = tl.arange(0, Dh)
    KV_offsets = offsets_COL_1[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    KV_mask = (offsets_COL_1[:, None] < N) # to avoid out-of-bounds non-existent tokens
    K = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.) # shape (BLOCK_SIZE_COL_1, Dh)
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.) # shape (BLOCK_SIZE_COL_1, Dh)

    # pre-scaling K allows us to do the multiplication once here as opposed to
    #  num_steps times inside _attn_backward_KV
    # we also scale by rln2 to account for the derivative of tl.exp2() and do it
    #  here instead of inside _attn_backward_KV for the same reason
    K *= scale * rln2

    # we'll accumulate the gradients into these
    dLdK = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)

    # compute dLdK and dLdV portions along the blocked diagonal
    dLdK, dLdV = _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    # next we'll do all the blocks that don't need the triangular mask on the block-diagonal.
    # this moves us forward to get off of the block-diagonal
    start_ROW += BLOCK_SIZE_COL_1
        # start_COL doesn't change since that's determined by our PID
    # then we calculate how many blocks need to be done. 
    # this adjustment to N accounts for sequence lengths that are not clean multiples of BLOCK_SIZE_COL_1
    N_adj = tl.cdiv(N, BLOCK_SIZE_COL_1) * BLOCK_SIZE_COL_1
    num_steps = (N_adj - start_ROW) // BLOCK_SIZE_ROW_1

    # compute dLdK and dLdV for non-masked blocks
    dLdK, dLdV = _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False ###
    )

    # scale since we didn't do it inside _attn_backward_KV to save flops
    dLdK *= scale * rln2
    # write back dLdK and dLdV
    tl.store(dLdK_ptr + KV_offsets, dLdK, mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV, mask=KV_mask)

    ### STAGE 2: Now we do dLdQ
    # in this part, like the forward pass we look at a specific block of Q & iterate through K & V

    # ROW and COL refer to the N dimension of Q and K/V respectively
    # for stage 1 each PID will look at BLOCK_SIZE_MACRO tokens of K & V
    # and there will be an inner for loop iterating over BLOCK_SIZE_MICRO tokens of Q
    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO

    # we again start off doing the block-diagonal
    start_ROW = pid * BLOCK_SIZE_ROW_2
    start_COL = start_ROW
    num_steps = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2
    # ^this is number of steps for a single block, aka the blocks on the diagonal

    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW_2)
    QO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    mask_ROW = offsets_ROW < N
    Q = tl.load(Q_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.) # shape (BLOCK_SIZE_ROW_2, Dh) 
    Q *= scale * rln2
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.) # shape (BLOCK_SIZE_ROW_2, Dh) 
    LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_ROW, other=0.)[:, None] # shape (BLOCK_SIZE_ROW_2, 1) 

    # accumulate the gradients into here
    dLdQ = tl.zeros([BLOCK_SIZE_ROW_2, Dh], dtype=tl.float32)

    # compute dQ for blocks on the diagonal
    dLdQ = _attn_backward_Q(
        dLdQ, Q, dLdO, LSE, 
        K_ptr, V_ptr, Delta_ptr, 
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    # now we'll do the parts that are not on the block-diagonal
    end_COL = start_COL
    start_COL = 0 #end_COL - num_steps * BLOCK_SIZE_COL_2 # could just call it 0 lmao
    num_steps = end_COL // BLOCK_SIZE_COL_2
    dLdQ = _attn_backward_Q(
        dLdQ, Q, dLdO, LSE, 
        K_ptr, V_ptr, Delta_ptr, 
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )
    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ, mask=mask_ROW[:, None])
    


class _flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k, v, scale):
        assert q.shape == k.shape == v.shape    
        assert q.shape[-1] <= 128
        assert q.device == k.device and k.device == v.device
        assert q.dtype == k.dtype == v.dtype == torch.float32

        B, H, N, Dh = q.shape

        # tensor to hold output
        O = torch.empty_like(q)

        # tensor to hold logsumexp
        LSE = torch.empty((B, H,N), device=q.device, dtype=torch.float32)

        grid = lambda args: (
            triton.cdiv(N, args["BLOCK_SIZE_QO"]),
            B * H 
        ) 
        """
        (3,2)
        3 SMs
        [0, 0 ] \   SM0
        [1, 0 ] /
        [2, 0 ] \   SM1
        [0, 1 ] /
        [1, 1 ] \   SM2
        [2, 1 ] /
        """

        attn_fwd[grid](
            q, k ,v, O, LSE,
            scale,
            q.stride(0),q.stride(1),q.stride(2),q.stride(3),
            k.stride(0),k.stride(1),k.stride(2),k.stride(3),
            v.stride(0),v.stride(1),v.stride(2),v.stride(3),
            O.stride(0),O.stride(1),O.stride(2),O.stride(3),
            LSE.stride(0),LSE.stride(1),LSE.stride(2),
            B, H, N, Dh,
        )

        ctx.save_for_backward(q,k,v, O, LSE)
        ctx.grid = grid
        ctx.B, ctx.H, ctx.N, ctx.Dh = B, H, N,Dh
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE = ctx.saved_tensors
        grid = ctx.grid
        scale = ctx.scale
        B, H, N,Dh = ctx.B, ctx.H, ctx.N, ctx.Dh

        dLdq = torch.empty_like(q)  # shape(B, H, N , Dh)
        dLdk = torch.empty_like(k)
        dLdv = torch.empty_like(v)

        dLdO = dLdO.contiguous()
        assert q.stride() == k.stride() == v.stride() == O.stride() == dLdO.stride()

        Delta = torch.empty_like(LSE) # shape ( B, H , N)

        pre_grid = lambda meta: (triton.cdiv(N, meta["PRE_BLOCK_SIZE_ROW"]), B * H)

        attn_backward_preprocess[pre_grid](
            O, dLdO, Delta,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dLdO.stride(0), dLdO.stride(1), dLdO.stride(2), dLdO.stride(3),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            N, Dh,
        )


        grid = lambda meta: (
            triton.cdiv(N, meta['BLOCK_SIZE_MACRO']), B*H
        )
        attn_backward[grid](
            q, k ,v,
            dLdO, dLdq,dLdk,dLdv,
            LSE, Delta, 
            scale,
            q.stride(0),q.stride(1),q.stride(2),q.stride(3),
            H, N , Dh
        )

        return dLdq, dLdk, dLdv, None
 

        
triton_attention = _flashattention.apply

def test_attn_kernel(B, H, N , Dh, device = DEVICE, atol=5e-3 ):
    # switch into float16 for homework
    q = torch.randn((B , H, N, Dh), dtype=torch.float32, device=device, requires_grad=True)
    k = torch.randn((B , H, N, Dh), dtype=torch.float32, device=device, requires_grad=True)
    v = torch.randn((B , H, N, Dh), dtype=torch.float32, device=device, requires_grad=True)

    scale = 1/ math.sqrt(Dh)
    tri_out = triton_attention(
        q, k , v, scale
    )
    ref_out = torch.nn.functional.scaled_dot_product_attention(q,k,v, is_causal=True)
    torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
    print("Passed")

    dLdy = 0.1 * torch.randn_like(q)
    tri_out.backward(dLdy, retain_graph=True)
    dLdq_tri, dLdk_tri, dLdv_tri = [_.grad.clone() for _ in [q, k ,v]]
    q.grad, k.grad, v.grad= None, None, None

    ref_out.backward(dLdy, retain_graph=True)
    dLdq_ref , dLdk_ref , dLdv_ref = [_.grad.clone() for _ in [q, k ,v]]
    q.grad, k.grad, v.grad= None, None, None
    
    torch.testing.assert_close(dLdq_tri, dLdq_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdk_tri, dLdk_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdv_tri, dLdv_ref, atol=1e-2, rtol=0)
    print("Backward Passed")



configs = []
for mode in ["fwd", "bwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"],
            x_vals=[512 * i for i in range(1, 17)], # LOWER IF YOU DON'T HAVE ENOUGH RAM
            line_arg="provider",
            line_vals=["torch", 'this_tutorial'],
            line_names=[
                "torch.nn.functional.scaled_dot_product_attention", 
                "This tutorial's implementation"
                ],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=f"attention-performance-{mode}",
            args={"mode": mode},
        ))

@triton.testing.perf_report(configs)
def bench_flash_attention(SEQ_LEN, mode, provider, device=DEVICE):
    assert mode in ["fwd"] #, "bwd"]
    dtype = torch.float32
    BATCH, N_HEADS = 32, 4 # LOWER THESE IF YOU DON'T HAVE ENOUGH RAM
    HEAD_DIM = 128 # AND THIS IF YOU DON"T HAVE ENOUGH SRAM
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1 / math.sqrt(HEAD_DIM)
    if provider == 'torch':
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    if provider == 'this_tutorial':
        fn = lambda: triton_attention(q, k, v, sm_scale)
    if mode == "bwd":
        O = fn()
        dLdO = torch.randn_like(O)
        fn = lambda: O.backward(dLdO, retain_graph=True)
    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul * 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)



if __name__ == "__main__":
    test_attn_kernel(1, 1, 128, 32) # without block masking
    test_attn_kernel(1, 1, 128, 64) # without block masking
    test_attn_kernel(1, 1, 128, 128) # without block masking
    test_attn_kernel(32, 8, 69, 128) # 

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        bench_flash_attention.run(save_path='.', print_data=True)
        