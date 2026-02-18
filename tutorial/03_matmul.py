"""
Docstring for triton_project.03_matmul
- autonotic performacne tuning 
- PID re-ordering fro improved SRAM sharing between PIDS
- multi-dimensional pointer arithmetic
- data types -= high precision accumulation
- triton interpreter for improved debugging

A @ B = C
(M , K) @ ( K, N ) = ( M, N)

for m in range(0, M):
    for n in range(0, N):
        acc= 0
        for k in range(0, K ):
            a_vec = A[m, :]
            b_vec = B[:, n ]
            c = dot(a_vec, b_vec)

        C[m,n] = c

A @ B = C
(M,K) @ (K,N) = (M,N)
for m in range( 0 , M , BLOCK_SIZE_M ):
    for n in range(0, N , BLOCK_SIZE_N):
        acc = tl.zeros(shape = ( BLOCK_SIZE_M, BLOCK_SIZE_N) , dtype=tl.float32)
        for k in range(0, K , BLOCK_SIZE_K:
            a = A[m: m +BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
            b = B[ k : k+BLOCK_SIZE_K, n: n +BLOCK_SIZE_N]
            acc = dot( a ,b)
        C[m:m+BLOCK_SIZE_M, n: n +BLOCK_SIZE_N] = c
"""



import torch
import triton 
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")




# import os 
# os.environ["TRITON_INTERPRET"] = 1

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128 , 'BLOCK_SIZE_N': 256 , 'BLOCK_SIZE_K': 64 , 'GROUP_SIZE': 8 }, num_stages =3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 256 , 'BLOCK_SIZE_K': 32 , 'GROUP_SIZE': 8 }, num_stages = 4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128 , 'BLOCK_SIZE_N': 128 , 'BLOCK_SIZE_K': 32 , 'GROUP_SIZE': 8 }, num_stages =4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32 , 'GROUP_SIZE': 8 }, num_stages =4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 128 , 'BLOCK_SIZE_K': 32 , 'GROUP_SIZE': 8 }, num_stages =4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128 , 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32 , 'GROUP_SIZE': 8 }, num_stages =4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32 , 'GROUP_SIZE': 8 }, num_stages =5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32 , 'GROUP_SIZE': 8 }, num_stages =5, num_warps=2)
]

@triton.autotune(configs = autotune_configs, key=['M','N','K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N , K,
    a_M_stride, b_K_stride, c_M_stride,
    a_K_stride, b_N_stride, c_N_stride,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE: tl.constexpr,
):
    """
    M = N = K = 8
    BLOCK_SIZE_M/K/N = 2
        [ 0 , 1,   2,   3]
        [ 4 , 5,   6,   7]
        [ 8 , 9,   10, 11]
        [ 12 , 13, 14, 15]
    pid 0 for this example

       A           @       B           =       C
    [x, x, x, x]        [x, _, _, _]        [0, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]

    PID = 1
       A           @       B           
    [x, x, x, x]        [x, _, _, _]        
    [_, _, _, _]        [x, _, _, _]        
    [_, _, _, _]        [x, _, _, _]       
    [_, _, _, _]        [x, _, _, _]       

    PID = 2
       A           @       B           =       C
    [x, x, x, x]        [_, x, _, _]        
    [_, _, _, _]        [_, x, _, _]        
    [_, _, _, _]        [_, x, _, _]        
    [_, _, _, _]        [_, x, _, _]        

    PID = 3
       A           @       B           =       C
    [x, x, x, x]        [_, _, x, _]        
    [_, _, _, _]        [_, _, x, _]        
    [_, _, _, _]        [_, _, x, _]        
    [_, _, _, _]        [_, _, x, _]        

    PID = 4
       A           @       B           =       C
    [x, x, x, x]        [_, _, _, x]        
    [_, _, _, _]        [_, _, _, x]        
    [_, _, _, _]        [_, _, _, x]        
    [_, _, _, _]        [_, _, _, x]        

    if all pids are in the same SM, the row in A only loaded once 
    
    or we can do 
        PID = 1
       A           @       B           
    [x, x, x, x]        [x, x, _, _]        
    [x, x, x, x]        [x, x, _, _]        
    [_, _, _, _]        [x, x, _, _]       
    [_, _, _, _]        [x, x, _, _]      

    but need new ordering
        [ 0 , 2,   4,   6]
        [ 1 , 3,   5,   7]
        [ 8 , 10,   12, 14]
        [ 9 , 11, 13, 15]
        
        [ 0 , 2,  |  4,   6]
        [ 1 , 3,  |  5,   7]
        [------------------]
        [ 8 , 10, |  12, 14]
        [ 9 , 11, | 13,  15]

    """
    pid = tl.program_id(0)

    num_pid_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE * num_pid_along_N
    group_id = pid // num_pid_in_group

    first_pid_in_group_along_M = group_id * GROUP_SIZE
    group_size_adj = min(num_pid_along_M - first_pid_in_group_along_M, GROUP_SIZE)

    pid_m = first_pid_in_group_along_M + ((pid % num_pid_in_group) % group_size_adj)
    pid_n = (pid % num_pid_in_group) // group_size_adj

    offset_M = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_N = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_K = tl.arange(0, BLOCK_SIZE_K)

    a_offsets = offset_M[:, None] * a_M_stride + offset_K[None, :] * a_K_stride
    """
    [:, None] turns [m1 , m2, m3 ] into [[m1], [m1], [m1]]
    [None, :] turns [m1 , m2, m3 ] into [[m1, m1, m1]
    
    [[m1n1, m1n2, m1n3],
     [m2n1, m2n2, m2n3],
     [m3n1, m3n2, m3n3]]
    """
    b_offsets = offset_K[:, None] * b_K_stride + offset_N[None, :] * b_N_stride

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offset_K < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptr + a_offsets, mask[None,:], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask[:, None], other=0.0)

        accumulator = tl.dot(a, b ,accumulator)
        a_offsets += BLOCK_SIZE_K * a_K_stride
        b_offsets += BLOCK_SIZE_K * b_K_stride
    
    accumulator = accumulator.to(tl.float16)
    c_ptrs = c_ptr + offset_M[:, None] * c_M_stride + offset_N[None, :] * c_N_stride
    mask = (offset_M[:, None] < M) & (offset_N[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

def matmul(a , b):

    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[0]

    (M,K) , (_,N) = a.shape, b.shape

    c = torch.empty((M,N), device=a.device, dtype=torch.float16)

    '''
        [ 0 , 1, 2, ,3]
        [ 4 , 5, 6, ,7]
        [ 8 , 9, 10, 11,]
        [ 12 , 13, 14, ,15]
        these are the pid 
    '''

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(M, meta['BLOCK_SIZE_N']),
    ) # 16 on above scenario

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0),b.stride(0),c.stride(0),
        a.stride(1),b.stride(1),c.stride(1),

    )
    return c

def test_matmul_kernel(size:list, atol=1e-3, rtol=1e-1, device=DEVICE):
    torch.manual_seed(0)
    assert type(size) == list and type(size[0]) == tuple and len(size) == 2
    a = torch.randn(size[0], device=DEVICE, dtype=torch.float16)
    b = torch.randn(size[1], device=DEVICE, dtype=torch.float16)

    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("Passed")


configs = [
    triton.testing.Benchmark(
        x_names = ["M","N","K"],
        x_vals = [128*i for i in range(2,33)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch", "Triton"],
        styles = [("green", "-"), ("blue","-")],
        ylabel="TFLOPS",
        plot_name = "matmul-performance",
        args={},
    )
]
@triton.testing.perf_report(configs)
def benchmark(M,N,K, provider):
    a = torch.randn((M,K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K,N), device=DEVICE, dtype=torch.float16)

    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms * 1e-3)
        # 3 = number of memory operations (2 read + 1 write)
        # M * N * K = number of elements per memory op
        # 1e-12 converts flops to Teraflops
        # 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    test_matmul_kernel([(175, 100) , (100, 23)])

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)