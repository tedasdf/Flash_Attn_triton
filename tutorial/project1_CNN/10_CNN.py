import torch
import triton
import triton.language as tl

from im2col import im2col_triton ,col2im_triton
from tutorial.project1_CNN.GEMM import grouped_matmul

class CNN_layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, padding, stride):
        # x: (B, C_in, H, W)
        # weight: (C_out, C_in, K_h, K_w)
        B, C_in, H, W = x.shape
        C_out, _, K_h, K_w = weight.shape

        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1

        # 1. Run im2col to get the "Big Ass Matrix"
        # Shape: (B * H_out * W_out, C_in * K_h * K_w)
        A_matrix = im2col_triton(x, K_h, K_w, stride, padding)

        # 2. Prepare the Weight Matrix
        # Shape: (C_in * K_h * K_w, C_out)
        B_matrix = weight.view(C_out, -1).t().contiguous()

        # 3. GEMM: C = A @ B
        # Output shape: (B * H_out * W_out, C_out)
        C_matrix = grouped_matmul([A_matrix], [B_matrix])[0]

        # 4. Reshape back to CNN format (B, C_out, H_out, W_out)
        # We need to permute because GEMM gives us (Patches, Channels)
        out = C_matrix.view(B, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()

        # Save for backward: We need the im2col matrix to calculate dLdw
        # and the weights to calculate dLx
        ctx.save_for_backward(A_matrix, weight)
        ctx.params = (padding, stride, x.shape) # Store shapes for reconstruction

        return out
 
    @staticmethod
    def backward(ctx, dLdy):
        # dLdy shape: (B, C_out, H_out, W_out)
        A_matrix, weight = ctx.saved_tensors
        padding, stride, x_shape = ctx.params
        
        # 1. Prepare dLdy for GEMM
        # Reshape (B, C_out, H_out, W_out) -> (B * H_out * W_out, C_out)
        dLdy_flat = dLdy.permute(0, 2, 3, 1).reshape(-1, dLdy.shape[1]).contiguous()

        # 2. Calculate dLdW = A.T @ dLdy
        # You can use your grouped_matmul here if you transpose A_matrix
        # dLdW_flat shape: (C_in * K_h * K_w, C_out)
        dLdW_flat = torch.matmul(A_matrix.t(), dLdy_flat) 
        dLdW = dLdW_flat.t().view(weight.shape)

        # 3. Calculate dLx (Gradient w.r.t Input)
        # dLx_flat = dLdy_flat @ Weight.T
        weight_flat = weight.view(weight.shape[0], -1) # (C_out, K_items)
        dLx_col = torch.matmul(dLdy_flat, weight_flat) # (Patches, K_items)
        
        # 4. Col2Im: Convert dLx_col back to (B, C, H, W)
        # This requires a new kernel (inverse of im2col)
        dLx = col2im_triton(dLx_col, x_shape, padding, stride)

        return dLx, dLdW, None, None, None