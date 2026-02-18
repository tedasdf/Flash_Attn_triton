"""
Docstring for triton_project.04_dropout
- parallel pseudo-random number generation in SRAM
"""

import torch
import triton 
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
 
@triton.jit
def _dropout_kernel(
    x_ptr, 
    output_ptr, 
    n_elements, 
    p, 
    seed ,
    BLOCK_SIZE:tl.constexpr
):
    pid = tl.program_id*axis=0
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets  < n_elements 

    x = tl.load(x_ptr + offsets , mask=mask, other=0.0)
    random = tl.rand(seed, offsets)
    # shape of offsets
    x_keep = random > p 
    output = tl.where(x_keep, x / (1-p) , 0 )

    tl.store(output_ptr + offsets , output, mask=mask)



def seeded_dropout(x, p , seed ):
   
    output = torch.empty_like(x)
    assert x.is_contiguous()

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _dropout_kernel[grid](
        x, output,
        n_elements, p, seed,
        BLOCK_SIZE=1024
    )

if __name__ == "__main__":
    x = torch.randn(size=(8,), device= DEVICE)
    output1 = seeded_dropout(x, p=0.5, seed=123)