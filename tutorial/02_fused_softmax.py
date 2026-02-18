"""
Docstring for triton_project.02_fused_softmax
- reduce memory reads/writes by fusin
-some more detials on GPU architecture
- how to define meta-parameters using heuristics and GPU-specific attritbutes
- more about masking and how to choose the value of extra masked out entries
"""


import torch
import triton 
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


def naive_softmax(x):
    # assume input size (M,N)

    # read MN element and write M elements
    x_max = x.max(dim=1)[0]

    # read M and MN elements and write MN
    z = x - x_max[:, None] # shape (M , N ) - shape (M, 1) = shape (M,N)
    
    # reading MN elements and writing MN elements
    numerator = torch.exp(z)            # shape (M, N)

    # read MN elements, then MN flops, write M elements 
    denomincator = numerator.sum(1)     # shape (M, N ) - >  shape( M)
    
    # read MN + M elements and write MN element
    out = numerator / denomincator[:, None] # shape (M,N) / shape (M,1) = shape(M,N)

    return out

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    # shape (M, N)
    # BLOCK_SIZE = next

    pid = tl.program_id(0)
    row_step = tl.num_programs(0)

    # if 4 programs, then row_step = 4
    # if n_rows= 6
    # pid 0 would get row 0
    # pid 1 would get row 1
    # pid 2 would get row 2
    # pid 3 would get row 3
    # once they're done with their first assigned rows
    # pid 0 += row_step would get row 4
    # pid 1 += row_step would get row 5
    
    for row_idx in tl.range(pid, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride

        col_offset = tl.arange(0, BLOCK_SIZE)   
        input_ptrs = row_start_ptr + col_offset  
        
        mask = col_offset < n_cols

        # n_cols = 3
        # BLOCKS_SIZE = next_power_of_2 ( n_cols ) =4 
        X_val = tl.load(input_ptrs, mask=mask, other=-float('inf')) # SHAPE (BLOCK_SIZE) which is roughly (n_cols)

        x_max = X_val - tl.max(X_val, axis=0)   # shape (BLOCK_SIZE) - (1) -> (BLOCK_SIZE)
        numerator = tl.exp(x_max)              # shape (BLOCK_SIZE)
        denominator = tl.sum(numerator, axis=0) # shape (1)
        Y_val = numerator / denominator         # shape (BLOCK_SIZE) / (1) -> (BLOCK_SIZE)
        

        output_ptrs = output_ptr + row_idx * output_row_stride
        tl.store(output_ptrs + col_offset , Y_val,  mask=mask)


# step 3
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"] 
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] # how many program can be put onto one SM
WARP_SIZE = properties["warpSize"]



def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    # every row fit within sm
    BLOCK_SIZE = triton.next_power_of_2(n_cols) 

    num_warps = 4 
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2
    
    y = torch.empty_like(x)


    # how many register and how much SRAM 
    kernel = _softmax_kernel.warmup(
        x,y,x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)# shape of launch grid
    )


    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared

    reg_occupany = NUM_REGS // (n_regs_per_program * WARP_SIZE * num_warps)
        # NUM_REGS = 65536
        # each program might use 
            # n_regs_per_program = 32
            # WARP_SIZE = 32
            # num_warps = 8
        # so each program needs ( n_regs_per_program * WARP_SIZE * num_warps) registers total 
    
    sram_occupany = TOTAL_SRAM_PER_SM // sram_needed_per_program
    program_per_sm = min(reg_occupany, sram_occupany)


    num_programs = min(NUM_SM * program_per_sm, n_rows)
    grid = (num_programs, 1, 1 )


    kernel[(num_programs, 1, 1)](x, y, x.stride(0), y.stride(0), n_rows, n_cols)
    return y
    # x is shape (M,N)
    # x.stride() would be (N,1)
    # x.stride(0) would be N
    # x.stride(1) would be 1
    # z shape (B,N,D)
    # z.tride() should be (N*D, D , 1)




def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    assert type(size) is tuple and len(size) == 2
    torch.manual_seed(0)
    x = torch.randn(size[0] , size[1], device=DEVICE)
    z_tri = softmax(x)
    z_ref = torch.softmax(x, axis=1)

    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("Passed")





@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green','-')],
        ylabel='GB/s',
        plot_name='softmax-performance',
        args={'M':4096}
    )
)
def benchmark(M, N, provider):
    x = torch.rand(M,N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)


    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda:  softmax(x))
    
    # 3                 number of memory operations 
    # x.numel()         nyumber of element
    # x.element_size()  dtype
    gbps = lambda ms: 2  * x.numel() * x.element_size() *1e-9 /(ms*1e-3)

    return gbps(ms)

if __name__ == "__main__":
    test_softmax_kernel((1823, 781))
    import sys 

    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark.run(save_path='.', print_data=True)