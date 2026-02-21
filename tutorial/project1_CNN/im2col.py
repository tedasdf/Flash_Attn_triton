import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


# Conceptual optimization for the inner loop
@triton.jit
def optimized_im2col_kernel(
    x_ptr, col_matrix_ptr,
    B, C, H, W,
    kh, kw,
    stride_h, stride_w,
    pad_h, pad_w,
    H_out, W_out,
    NUM_SM: tl.constexpr,
    BLOCK_PATCHES: tl.constexpr # How many rows of the matrix to do at once
):
    pid = tl.program_id(0)
    
    # Process a BLOCK of patches to increase reuse
    patch_offsets = tl.arange(0, BLOCK_PATCHES)
    curr_patches = pid * BLOCK_PATCHES + patch_offsets
    
    # Calculate coords for all patches in the block at once
    batch_idx = curr_patches // (H_out * W_out)
    # ... indexing math ...

    # Load a whole chunk of channels/kernel space at once
    # This allows the GPU to use 'coalesced' memory access
    cols = tl.arange(0, BLOCK_CHANNELS) 
    
    # x_ptr + offset now points to a block of data, not just one pixel
    data = tl.load(x_ptr + wide_offset, mask=mask)
    tl.store(col_ptr + out_wide_offset, data)


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


@triton.jit
def col2im_kernel(
    col_ptr,      # The dLx_col matrix (Patches x K_items)
    out_ptr,      # The dLx image (B x C x H x W)
    B, C, H, W,
    K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    out_h, out_w,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a block of elements from the col_matrix
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # col_matrix shape is (TotalPatches, C * Kh * Kw)
    # TotalPatches = B * out_h * out_w
    num_patches = B * out_h * out_w
    K_items = C * K_h * K_w
    
    # 1. Map the linear index to (patch_idx, channel_kernel_idx)
    patch_idx = idx // K_items
    ck_idx = idx % K_items
    
    # 2. Map patch_idx to (b, oh, ow)
    b = patch_idx // (out_h * out_w)
    rem = patch_idx % (out_h * out_w)
    oh = rem // out_w
    ow = rem % out_w
    
    # 3. Map ck_idx to (channel, kh, kw)
    c = ck_idx // (K_h * K_w)
    rem_k = ck_idx % (K_h * K_w)
    kh = rem_k // K_w
    kw = rem_k % K_w
    
    # 4. Calculate the target pixel in the original image
    h_in = oh * stride_h - pad_h + kh
    w_in = ow * stride_w - pad_w + kw
    
    # 5. Check if the pixel is within bounds (not padding)
    mask = (patch_idx < num_patches) & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
    
    # 6. Compute memory address in the dLx image
    # Shape: (B, C, H, W)
    out_offset = (b * C * H * W + 
                  c * H * W + 
                  h_in * W + 
                  w_in)
    
    # Load the gradient value from the col_matrix
    grad_val = tl.load(col_ptr + idx, mask=(patch_idx < num_patches))
    
    # 7. ATOMIC ADD: Multiple patches contribute to the same pixel
    tl.atomic_add(out_ptr + out_offset, grad_val, mask=mask)


def test_im2col(B, C_in, C_out, H, W, kernel_h, kernel_w, padding, stride):
    print(f"Testing Conv Layer: {H}x{W} -> Filters: {C_out}")

    # 1. Initialize Tensors (Corrected shapes)
    x = torch.randn((B, C_in, H, W), device=DEVICE, dtype=torch.float32)
    weight = torch.randn((C_out, C_in, kernel_h, kernel_w), device=DEVICE, dtype=torch.float32)

    # 2. Reference Output (PyTorch)
    ref_out = torch.nn.functional.conv2d(x, weight, stride=stride, padding=padding)
    H_ref, W_ref = ref_out.shape[2], ref_out.shape[3]

    # 3. Triton Pipeline
    # A. Run im2col
    col_matrix, H_out, W_out = im2col_triton(x, kernel_h, kernel_w, stride, padding)
    
    # B. Reshape Weights for Matrix Mult (K, N) where K = flattened kernel
    # weight is (C_out, C_in, Kh, Kw) -> we want (C_in * Kh * Kw, C_out)
    weight_matrix = weight.view(C_out, -1).t().contiguous()

    # C. Perform GEMM (using simple torch.mm or your grouped_matmul)
    # Result shape: (B * H_out * W_out, C_out)
    out_flat = torch.matmul(col_matrix, weight_matrix)

    # D. Reshape result back to 4D (B, C_out, H_out, W_out)
    # The GEMM gives us (B, H_out, W_out, C_out). We permute to get Channels to dim 1.
    out_final = out_flat.view(B, H_out, W_out, C_out).permute(0, 3, 1, 2)

    # 4. Verification
    try:
        torch.testing.assert_close(out_final, ref_out, atol=1e-3, rtol=1e-3)
        print("✅ Success! Triton output matches PyTorch.")
    except Exception as e:
        print("❌ Failure! Outputs do not match.")
        print(e)



def test_col2im(B, C_in, C_out, H, W, kernel_h, kernel_w, padding, stride):
    print(f"Testing Conv LAyer: {H} X {W} -> Filters: {C_out}")

    x = torch.randn((B, C_in, H, W), device=DEVICE, dtype=torch.float32)
    weight = torch.randn((C_out, C_in, kernel_h, kernel_w), device=DEVICE, dtype=torch.float32)
    
    ref_out = torch.nn.functional.conv2d(x, weight, stride=stride, padding=padding)
    H_ref, W_ref = ref_out.shape[2], ref_out.shape[3]

    col_matrix, H_out, W_out = im2col_triton(x, kernel_h, kernel_w, stride, padding)
    weight_matrix = weight.view(C_out, -1).t().contiguous()

    out_flat = torch.matmul(col_matrix, weight_matrix)

    out_final = out_flat.view(B, H_out, W_out, C_out).permute(0,3,1,2)

    try:
        torch.testing.assert_close(out_final, ref_out, atol=1e-3, rtol=1e-3)
        print("✅ Success! Triton output matches PyTorch.")
    except Exception as e:
        print("❌ Failure! Outputs do not match.")
        print(e)
        
if __name__ == "__main__":
    # Test with standard 3x3 convolution settings
    test_im2col(B=2, C_in=3, C_out=16, H=32, W=32, 
                kernel_h=3, kernel_w=3, padding=1, stride=1)