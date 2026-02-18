import torch 

import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


@triton.jit
def add_kernel(
        x_ptr, 
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
    pid = tl.program_id(0)
    # vec of length 256
    # BLOCK_SIZE 64
    # PID 0 might process elements [0:64]
    # PID 1 might process elements [65:128]
    
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask =  offset < n_elements
    


    # load data from DRAM/ VARM / HBM to SRAM/on-chip memory
    X_Val = tl.load(x_ptr + offset , mask, other=None)
    Y_val = tl.load(y_ptr + offset, mask, other=None)
    
    output = X_Val + Y_val
    tl.store(output_ptr + offset, output, mask=mask)


def add(x,y):
    output = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE

    n_elements = output.numel() # give the total entry of the tensor 
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    # cdiv(m, n) = ( m + ( n - 1)) // n 

    add_kernel[grid](
        x, 
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output

    

def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size, device= DEVICE)
    y = torch.randn(size, device= DEVICE)

    z_tri = add(x,y)
    z_ref = x + y

    torch.testing.assert_close(z_tri , z_ref, atol=atol, rtol=rtol)
    print("Passed")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green','-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={}
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x+ y , quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x,y) , quantiles=quantiles)
    
    # 3                 number of memory operations 
    # x.numel()         nyumber of element
    # x.element_size()  dtype
    gbps = lambda ms: 3  * x.numel() * x.element_size()

    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=98432)
    test_add_kernel(size=4097)

    import sys 

    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark.run(save_path='.', print_data=True)