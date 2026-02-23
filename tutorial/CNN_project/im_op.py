import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


@triton.jit
def im2col_kernel(
    x_ptr, col_matrix_ptr,
    B, C, H, W,
    kh, kw,
    stride_h, stride_w,
    pad_h, pad_w,
    H_out, W_out,
    NUM_SM: tl.constexpr,
    BLOCK_PATCHES: tl.constexpr # How many rows of the matrix to do at once
):
    # Each program (SM) starts at its ID and jumps by NUM_SM
    pid = tl.program_id(0)
    
    # Total rows in our big matrix
    total_patches = B * H_out * W_out
    # Total columns (the flattened kernel)
    patch_size = C * kh * kw

    for row_idx in range(pid, total_patches, NUM_SM):
        # 1. Map row_idx to image coordinates
        batch_idx = row_idx // (H_out * W_out)
        out_idx = row_idx % (H_out * W_out)
        curr_out_h = out_idx // W_out
        curr_out_w = out_idx % W_out

        # 2. Calculate the top-left corner in the input image
        in_h_start = curr_out_h * stride_h - pad_h
        in_w_start = curr_out_w * stride_w - pad_w

        # 3. Iterate through Channels and Kernel size to fill the columns
        for c in range(C):
            for i in range(kh):
                for j in range(kw):
                    h_in = in_h_start + i
                    w_in = in_w_start + j
                    
                    # Padding logic: Check if we are inside image bounds
                    is_valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                    
                    # Calculate pointers
                    # Input: (batch, channel, height, width)
                    in_offset = (batch_idx * C * H * W + 
                                 c * H * W + 
                                 h_in * W + 
                                 w_in)
                    
                    # Output Matrix: (row_idx, col_idx)
                    # col_idx is our flattened (c, i, j)
                    col_idx = c * (kh * kw) + i * kw + j
                    out_offset = row_idx * patch_size + col_idx
                    
                    # Load pixel or 0.0 if padding
                    pixel = tl.load(x_ptr + in_offset, mask=is_valid, other=0.0)
                    tl.store(col_matrix_ptr + out_offset, pixel)

@triton.jit
def col2im_kernel(
    x_ptr,           # (Batch, C * Kh * Kw, L)
    out_ptr,         # (Batch, C, H, W)
    # Dimensions
    batch, channel, h_out, w_out,
    kernel_h, kernel_w,
    stride, padding,
    h_blocks, w_blocks,
    # Strides (from x.stride())
    stride_xb, stride_xr, stride_xl,
    # Strides (from out.stride())
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 1. Identify which pixels this program handles
    outpix_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = outpix_offsets < (batch * channel * h_out * w_out)
    
    # 2. Un-flatten into 4D coordinates
    out_w_idx = outpix_offsets % w_out
    out_h_idx = (outpix_offsets // w_out) % h_out
    out_c_idx = (outpix_offsets // (w_out * h_out)) % channel
    out_b_idx = (outpix_offsets // (w_out * h_out * channel))

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 3. Iterate through every position in the kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_pad = out_h_idx + padding
            w_pad = out_w_idx + padding

            # Check: Did the kernel sliding window actually land on this pixel?
            # Must use & (bitwise) for vectors in Triton, not 'and'
            is_valid_stride = ((h_pad - kh) % stride == 0) & ((w_pad - kw) % stride == 0)
            
            # Use integer division //
            i = (h_pad - kh) // stride
            j = (w_pad - kw) // stride

            # Check: Is the calculated patch (i, j) within the valid forward grid?
            is_valid_window = is_valid_stride & (i >= 0) & (i < h_blocks) & (j >= 0) & (j < w_blocks)
            
            # 4. Calculate memory locations in the Col Matrix (x)
            row_idx = out_c_idx * (kernel_h * kernel_w) + kh * kernel_w + kw
            l_idx = i * w_blocks + j
            
            x_offsets = (out_b_idx * stride_xb) + (row_idx * stride_xr) + (l_idx * stride_xl)
            
            # Load and accumulate - mask handles both out-of-bounds pixels and invalid windows
            # 
            contribution = tl.load(x_ptr + x_offsets, mask=mask & is_valid_window, other=0.0)
            acc += contribution

    # 5. Final Store: Write the accumulated gradient back to the image
    out_offsets = (out_b_idx * stride_ob) + (out_c_idx * stride_oc) + \
                  (out_h_idx * stride_oh) + (out_w_idx * stride_ow)
    
    tl.store(out_ptr + out_offsets, acc, mask=mask)


def im2col_triton(x, kernel_h, kernel_w, stride, padding):
    B, C, H, W = x.shape  # Fixed: shape is an attribute

    # Standard formula for output dimensions
    H_out = (H + 2 * padding - kernel_h) // stride + 1
    W_out = (W + 2 * padding - kernel_w) // stride + 1

    # Get device properties for the grid
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    
    # Allocate the 'Big Ass Matrix'
    # Rows: Total number of patches across batch and space
    # Cols: Flattened pixels in one patch (C * Kh * Kw)
    col_matrix = torch.empty((B * H_out * W_out, C * kernel_h * kernel_w), 
                             device=x.device, dtype=x.dtype)

    # Launch kernel with your persistent grid strategy
    grid = (num_sms,)
    im2col_kernel[grid](
        x, col_matrix,
        B, C, H, W,
        kernel_h, kernel_w,
        stride, stride,
        padding, padding,
        H_out, W_out,
        NUM_SM=num_sms,
        BLOCK_PATCHES=10 # Start with 1, can be tuned
    )

    return col_matrix, H_out, W_out 

def col2im_triton(x, kernel_h, kernel_w, stride, padding, h_out, w_out, channels):
    # x is expected to be (Batch, C * Kh * Kw, L)
    # If your GEMM gives (B*L, C*Kh*Kw), you must reshape it first!
    B, C_Kh_Kw, L = x.shape
    
    # Calculate how many windows there were in the forward pass
    h_blocks = (h_out + 2 * padding - kernel_h) // stride + 1
    w_blocks = (w_out + 2 * padding - kernel_w) // stride + 1

    out_images = torch.zeros((B, channels, h_out, w_out), 
                            device=x.device, dtype=x.dtype)

    total_pixels = B * channels * h_out * w_out
    BLOCK_SIZE = 1024 
    grid = lambda meta: (triton.cdiv(total_pixels, meta['BLOCK_SIZE']),)

    col2im_kernel[grid](
        x, out_images,
        B, channels, h_out, w_out,
        kernel_h, kernel_w,
        stride, padding,
        h_blocks, w_blocks,
        # Pass the strides so Triton knows how to navigate the memory
        x.stride(0), x.stride(1), x.stride(2),
        out_images.stride(0), out_images.stride(1), out_images.stride(2), out_images.stride(3),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out_images

def test_im2col_and_col2im(B, C_in, C_out, H, W, kernel_h, kernel_w, padding, stride):
    print(f"Testing Pipeline: {H}x{W} -> Filters: {C_out}")

    # --- FORWARD PASS TEST ---
    x = torch.randn((B, C_in, H, W), device=DEVICE, dtype=torch.float32)
    weight = torch.randn((C_out, C_in, kernel_h, kernel_w), device=DEVICE, dtype=torch.float32)

    # 1. PyTorch Reference
    ref_out = torch.nn.functional.conv2d(x, weight, stride=stride, padding=padding)
    
    # 2. Triton Forward
    col_matrix, H_out, W_out = im2col_triton(x, kernel_h, kernel_w, stride, padding)
    weight_matrix = weight.view(C_out, -1).t().contiguous()
    out_flat = torch.matmul(col_matrix, weight_matrix)
    out_final = out_flat.view(B, H_out, W_out, C_out).permute(0, 3, 1, 2)

    torch.testing.assert_close(out_final, ref_out, atol=1e-3, rtol=1e-3)
    print("✅ Forward (im2col) Success!")

    # --- BACKWARD PASS TEST (col2im) ---
    # Imagine dL/dy (gradient of output) is out_final
    grad_output = torch.randn_like(out_final) # (B, C_out, H_out, W_out)
    
    # In backward, we multiply grad_output by weight transposed to get dL/d_cols
    # Shape: (B * H_out * W_out, C_in * kernel_h * kernel_w)
    grad_output_flat = grad_output.permute(0, 2, 3, 1).reshape(-1, C_out)
    grad_col = torch.matmul(grad_output_flat, weight_matrix.t()) 
    
    # Reshape for col2im: (B, C_in * Kh * Kw, L)
    # Note: L = H_out * W_out
    grad_col = grad_col.view(B, H_out * W_out, -1).permute(0, 2, 1).contiguous()

    # 3. Triton col2im
    # This should reconstruct the gradient back to the shape of 'x' (B, C_in, H, W)
    tri_grad_input = col2im_triton(
        grad_col, kernel_h, kernel_w, stride, padding, 
        h_out=H, w_out=W, channels=C_in
    )

    # 4. PyTorch Reference for col2im (using Fold)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=(kernel_h, kernel_w), 
                         padding=padding, stride=stride)
    ref_grad_input = fold(grad_col)

    torch.testing.assert_close(tri_grad_input, ref_grad_input, atol=1e-3, rtol=1e-3)
    print("✅ Backward (col2im) Success!")


if __name__ == "__main__":


    # Test with standard 3x3 convolution settings
    test_im2col_and_col2im(B=2, C_in=3, C_out=16, H=32, W=32, 
                kernel_h=3, kernel_w=3, padding=1, stride=1)