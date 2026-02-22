
import torch
from im_op import im2col_triton
from gemm import grouped_matmul


class CNN_layer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, weight, bias, padding ,stride
    ):  

        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(x, weight, bias)
        
        C_out , C_in , K_h, K_w = weight.shape()
        A_matrix,  H_out, W_out  = im2col_triton(x, K_h, K_w, stride, padding)
        
        w_flat = weight.view(weight.shape[0], -1).t()

        out_flat = grouped_matmul([A_matrix], [w_flat])
        
        # 4. Reshape to (B, C_out, H_out, W_out)
        output = out_flat.view(x.shape[0], H_out, W_out, -1).permute(0, 3, 1, 2)

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        return output

   

def run_test():
    # Setup parameters
    B, C_in, C_out = 2, 3, 8
    H, W = 32, 32
    K, S, P = 3, 1, 1
    device = torch.device("cuda")

    # Initialize data
    x = torch.randn((B, C_in, H, W), device=device)
    weight = torch.randn((C_out, C_in, K, K), device=device)
    bias = torch.randn(C_out, device=device)

    # 1. PyTorch Reference
    ref_out = torch.nn.functional.conv2d(x, weight, bias, stride=S, padding=P)

    # 2. Triton Implementation
    # Using .apply() to invoke the autograd function
    tri_out = CNN_layer.apply(x, weight, bias, P, S)

    # 3. Check equality
    try:
        torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)
        print("🚀 [SUCCESS]: Triton Forward matches PyTorch!")
    except Exception as e:
        print("❌ [FAILURE]: Mismatch detected.")
        print(e)

if __name__ == "__main__":
    run_test()