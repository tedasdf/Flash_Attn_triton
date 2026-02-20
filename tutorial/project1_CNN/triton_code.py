

import triton 
import torch 
import triton.language as tl
    

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")



def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"



def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_M_SIZE': 128,
            'BLOCK_N_SIZE': 128,
            'BLOCK_K_SIZE': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_M_SIZE': 128,
            'BLOCK_N_SIZE': 128,
            'BLOCK_K_SIZE': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_M_SIZE': 64,
            'BLOCK_N_SIZE': 64,
            'BLOCK_K_SIZE': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_M_SIZE': 64,
            'BLOCK_N_SIZE': 64,
            'BLOCK_K_SIZE': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_M_SIZE': 128,
            'BLOCK_N_SIZE': 128,
            'BLOCK_K_SIZE': 64,
            'NUM_SM': num_sms(),
        }),
        triton.Config({
            'BLOCK_M_SIZE': 64,
            'BLOCK_N_SIZE': 128,
            'BLOCK_K_SIZE': 64,
            'NUM_SM': num_sms(),
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    A_ptr_ptr, B_ptr_ptr, C_ptr_ptr,
    g_size_ptr, g_stride_ptr, group_num,
    NUM_SM: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr, BLOCK_N_SIZE: tl.constexpr, BLOCK_K_SIZE: tl.constexpr,
):
    global_tile_idx = tl.program_id(0)
    num_last_end = 0

    for g in range(group_num):
        M = tl.load(g_size_ptr + g * 3)
        N = tl.load(g_size_ptr + g * 3 + 1)
        K = tl.load(g_size_ptr + g * 3 + 2)

        A_stride = tl.load(g_stride_ptr + g * 3)
        B_stride = tl.load(g_stride_ptr + g * 3 + 1)
        C_stride = tl.load(g_stride_ptr + g * 3 + 2)

        A_ptr = tl.load(A_ptr_ptr + g ).to(tl.pointer_type(tl.float32))
        B_ptr = tl.load(B_ptr_ptr + g ).to(tl.pointer_type(tl.float32))
        C_ptr = tl.load(C_ptr_ptr + g ).to(tl.pointer_type(tl.float32))

        
        num_tiles_across_M = tl.cdiv(M, BLOCK_M_SIZE)
        num_tiles_across_N = tl.cdiv(N, BLOCK_N_SIZE)

        num_group_tiles = num_tiles_across_M * num_tiles_across_N


        while (global_tile_idx < num_last_end + num_group_tiles):
            if global_tile_idx >= num_last_end:
                local_tile_id = global_tile_idx - num_last_end
                tile_M_idx = local_tile_id // num_tiles_across_N
                tile_N_idx = local_tile_id % num_tiles_across_N

                # Initialize the block pointers at the start of the row/column for this tile
                a_block_ptr = tl.make_block_ptr(
                    base=A_ptr,
                    shape=(M, K),
                    strides=(A_stride, 1),
                    offsets=(tile_M_idx * BLOCK_M_SIZE, 0), # Start at K=0
                    block_shape=(BLOCK_M_SIZE, BLOCK_K_SIZE),
                    order=(0, 1)
                )
                b_block_ptr = tl.make_block_ptr(
                    base=B_ptr,
                    shape=(K, N),
                    strides=(B_stride, 1),
                    offsets=(0, tile_N_idx * BLOCK_N_SIZE), # Start at K=0
                    block_shape=(BLOCK_K_SIZE, BLOCK_N_SIZE),
                    order=(1, 0)
                )

                accumulator = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)

                for k in range(0, tl.cdiv(K, BLOCK_K_SIZE)):
                    a = tl.load(a_block_ptr, boundary_check=(0, 1))
                    b = tl.load(b_block_ptr, boundary_check=(0, 1))
                    
                    a_block_ptr = tl.advance(a_block_ptr, [0, BLOCK_K_SIZE])
                    b_block_ptr = tl.advance(b_block_ptr, [BLOCK_K_SIZE, 0])

                    accumulator = tl.dot(a, b, accumulator)

                c_block_ptr = tl.make_block_ptr(
                    base=C_ptr,
                    shape=(M, N),
                    strides=(C_stride, 1),
                    offsets=(tile_M_idx * BLOCK_M_SIZE, tile_N_idx * BLOCK_N_SIZE),
                    block_shape=(BLOCK_M_SIZE, BLOCK_N_SIZE),
                    order=(1, 0)
                )
                tl.store(c_block_ptr, accumulator.to(C_ptr.dtype.element_ty), boundary_check=(0, 1))

            global_tile_idx += NUM_SM
        num_last_end += num_group_tiles

def grouped_matmul(group_A, group_B):
    device = group_A[0].device
    A_ptrs = []
    B_ptrs = []
    C_ptrs = []
    group_sizes = []
    group_strides = []
    
    results = []

    for A, B in zip(group_A, group_B):
        M, K = A.shape
        K_verify, N = B.shape
        

        assert K == K_verify
        # Initialize output C
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        results.append(C)

        # Store raw memory addresses (pointers)
        A_ptrs.append(A.data_ptr())
        B_ptrs.append(B.data_ptr())
        C_ptrs.append(C.data_ptr())

        group_sizes.append((M, N, K))
        group_strides.append((A.stride(0),B.stride(0),C.stride(0)))

    A_ptr_tensor = torch.tensor(A_ptrs)
    B_ptr_tensor = torch.tensor(B_ptrs)
    C_ptr_tensor = torch.tensor(C_ptrs)
    g_stride_tensor = torch.tensor(group_strides)
    g_size_tensor = torch.tensor(group_sizes)

    grid = lambda meta: (
        meta['num_SM'],
    )

    grouped_matmul_kernel[grid](
        A_ptr_tensor, B_ptr_tensor, C_ptr_tensor,
        g_size_tensor, g_stride_tensor, g_size_tensor.shape[0]
    )

    return results

def test_GEMM(M_list, N_list, K_list, atol=1e-5, rtol=1e-3):
    print(f"Testing {len(M_list)} groups on {DEVICE}...")
    
    A_group = []
    B_group = []
    
    # Generate data
    for m, n, k in zip(M_list, N_list, K_list):
        A_group.append(torch.randn((m, k), device=DEVICE, dtype=torch.float32))
        B_group.append(torch.randn((k, n), device=DEVICE, dtype=torch.float32))

    # 1. Reference Output
    ref_out = [torch.matmul(a, b) for a, b in zip(A_group, B_group)]

    # 2. Triton Output 
    tri_out = grouped_matmul(A_group, B_group) 

    # 3. Verification
    for i in range(len(M_list)):
        try:
            torch.testing.assert_close(tri_out[i], ref_out[i], atol=atol, rtol=rtol)
            print(f"  ✅ Group {i} ({M_list[i]}x{N_list[i]}): Passed")
        except Exception as e:
            print(f"  ❌ Group {i} ({M_list[i]}x{N_list[i]}): Failed")
            raise e
    
    print("\n⭐ All tests passed!")


if __name__ == "__main__":
    ms = [1024, 453, 123, 64]
    ns = [233, 706, 100, 64]
    ks = [245, 851, 233, 32]
    
    test_GEMM(ms, ns, ks)