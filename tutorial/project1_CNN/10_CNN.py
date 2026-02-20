import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

@triton.jit
def im2col_triton(
    x_ptr, K_h, K_w, 
    stride, padding
):

def test_im2col(
        B, C_in, C_out, H, W,
        kernel_h, kernel_w, 
        padding, stride,
    ):
    
    H_out = ( H * 2 * padding - kernel_h) // (stride + 1)
    W_out = ( W * 2 * padding - kernel_w) // (stride + 1)

    x = torch.randn((B , C_in, H_out, W_out), device=DEVICE, dtype=torch.float32)
    weight = torch.randn((C_out, C_in , H_out, W_out), device=DEVICE, dtype=torch.float32)


    ref_out = torch.nn.functional.con2d(x, weight, stride=stride, padding=padding)

    weight_matrix = weight.view(C_out, -1).T().contiguous()
