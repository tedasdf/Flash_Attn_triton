
import torch
import torch.nn as nn
from im_op import col2im_triton, im2col_triton
from matmul import matmul
import math
import matplotlib.pyplot as plt

class CNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, padding, stride):
        B, C_in, H, W = x.shape
        C_out, _, Kh, Kw = weight.shape

        # 1. Forward im2col + Alignment Padding
        A_raw, H_out, W_out = im2col_triton(x, Kh, Kw, stride, padding)
        
        K_raw = A_raw.shape[1]
        K_padded = ((K_raw + 15) // 16) * 16 
        
        A_matrix = torch.zeros((A_raw.shape[0], K_padded), device=x.device, dtype=x.dtype)
        A_matrix[:, :K_raw] = A_raw

        w_flat_raw = weight.view(C_out, -1)
        w_padded = torch.zeros((C_out, K_padded), device=weight.device, dtype=weight.dtype)
        w_padded[:, :K_raw] = w_flat_raw
        w_flat = w_padded.t().contiguous()

        # 2. Forward GEMM
        out_flat = matmul(A_matrix, w_flat)
        output = out_flat.view(B, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        
        # Save necessary tensors for backward
        ctx.save_for_backward(x, weight, bias, A_matrix)
        ctx.padding, ctx.stride, ctx.K_raw = padding, stride, K_raw

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, A_matrix = ctx.saved_tensors
        padding, stride, K_raw = ctx.padding, ctx.stride, ctx.K_raw
        
        B, C_in, H, W = x.shape
        C_out, _, Kh, Kw = weight.shape
        _, _, H_out, W_out = grad_output.shape

        # --- 1. grad_bias ---
        grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None

        # --- 2. grad_weight (dw) ---
        # Flatten grad_output to (M, N) for GEMM
        # M = B * H_out * W_out, N = C_out
        grad_output_flat = grad_output.permute(0, 2, 3, 1).reshape(-1, C_out).contiguous().to(torch.float32)
        # dw = A_matrix.T @ grad_output_flat
        # Result is (K_padded, C_out)
        grad_w_padded = matmul(A_matrix.t().contiguous(), grad_output_flat)
        
        # Un-pad and reshape to original (C_out, C_in, Kh, Kw)
        grad_weight = grad_w_padded[:K_raw, :].t().contiguous().view(C_out, C_in, Kh, Kw)

        # --- 3. grad_input (dx) ---
        # We need grad_output_flat @ weight.T
        w_flat_raw = weight.view(C_out, -1)
        w_padded = torch.zeros((C_out, A_matrix.shape[1]), device=weight.device, dtype=weight.dtype)
        w_padded[:, :K_raw] = w_flat_raw
        
        # grad_A = grad_output_flat @ w_padded (transposed internally by your matmul if needed, 
        # but here we manually align: (M, C_out) @ (C_out, K_padded) = (M, K_padded)
        grad_A_padded = matmul(grad_output_flat, w_padded)
        grad_A_raw = grad_A_padded[:, :K_raw].contiguous()
        
        L = H_out * W_out
        grad_A_for_col2im = grad_A_raw.view(B, L, K_raw).permute(0, 2, 1).contiguous()

        # 4. Use your col2im to transform patches back to image gradient
        grad_input = col2im_triton(grad_A_for_col2im, Kh, Kw, stride, padding, H, W,  C_in)

        return grad_input, grad_weight, grad_bias, None, None


def visualize_tensors(ref_out, tri_out):
    # Move to CPU for plotting
    ref = ref_out.detach().cpu().numpy()
    tri = tri_out.detach().cpu().numpy()
    
    # We'll plot the first 3 channels of the first batch item
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for c in range(3):
        # Reference Row
        im0 = axes[0, c].imshow(ref[0, c], cmap='viridis')
        axes[0, c].set_title(f"Ref Channel {c}")
        plt.colorbar(im0, ax=axes[0, c])
        
        # Triton Row
        im1 = axes[1, c].imshow(tri[0, c], cmap='viridis')
        axes[1, c].set_title(f"Triton Channel {c}")
        plt.colorbar(im1, ax=axes[1, c])
        
    plt.tight_layout()
    plt.show()
   
def run_test():
    B, C_in, C_out = 2, 3, 8
    H, W = 32, 32
    K, S, P = 3, 1, 1
    device = torch.device("cuda")

    x = torch.randn((B, C_in, H, W), device=device, dtype=torch.float16)
    weight = torch.randn((C_out, C_in, K, K), device=device, dtype=torch.float16)
    bias = torch.randn(C_out, device=device, dtype=torch.float16)

    # --- 1. PREPARE REFERENCES ---
    ref_conv = torch.nn.functional.conv2d(x, weight, bias, stride=S, padding=P)
  
    # --- 2. RUN TRITON ---
    # Unpacks the 3 values returned by forward
    tri_conv = CNN.apply(x, weight, bias, P, S)
    
    # --- 3. CHECK im2col ---
    try:
        torch.testing.assert_close(tri_conv.float(), ref_conv.float(), atol=1e-2, rtol=1e-2)
        print("🚀 [SUCCESS]: Final Convolution matches!")
    except Exception as e:
        print("❌ [FAILURE]: Final output mismatch.")
        print(e)




class Conv2d_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # Initialize Weights (Kaiming Uniform is standard for CNNs)
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # We call the autograd function you defined earlier
        # Note: We pass padding and stride as constants
        out = CNN.apply(x, self.weight, self.bias, self.padding, self.stride)
        
        # Add bias (standard broadcasting: B, C, H, W + 1, C, 1, 1)
        return out + self.bias.view(1, -1, 1, 1)
    



if __name__ == "__main__":
    run_test()
