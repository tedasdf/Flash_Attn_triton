

import triton 
import torch 
import triton.language as tl
    
import matplotlib.pyplot as plt
import numpy as np
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")



def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"



def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_M_SIZE': 128,
            'BLOCK_N_SIZE': 128,
            'BLOCK_K_SIZE': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_M_SIZE': 128,
            'BLOCK_N_SIZE': 128,
            'BLOCK_K_SIZE': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_M_SIZE': 64,
            'BLOCK_N_SIZE': 64,
            'BLOCK_K_SIZE': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_M_SIZE': 64,
            'BLOCK_N_SIZE': 64,
            'BLOCK_K_SIZE': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_M_SIZE': 128,
            'BLOCK_N_SIZE': 128,
            'BLOCK_K_SIZE': 64,
            'NUM_SM': num_sms(),
        }),
        triton.Config({
            'BLOCK_M_SIZE': 64,
            'BLOCK_N_SIZE': 128,
            'BLOCK_K_SIZE': 64,
            'NUM_SM': num_sms(),
        }),
    ],
    key=['group_num'],
)
@triton.jit
def grouped_matmul_kernel(
    A_ptr_ptr, B_ptr_ptr, C_ptr_ptr,
    g_size_ptr, g_stride_ptr, group_num: tl.constexpr,
    NUM_SM: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr, BLOCK_N_SIZE: tl.constexpr, BLOCK_K_SIZE: tl.constexpr,
):
    global_tile_idx = tl.program_id(0)
    # The most direct way to get an int64 scalar 0
    num_last_end = tl.zeros((), dtype=tl.int64)

    for g in range(group_num):
        M = tl.load(g_size_ptr + g * 3)
        N = tl.load(g_size_ptr + g * 3 + 1)
        K = tl.load(g_size_ptr + g * 3 + 2)

        # Load group-specific pointers
        A_ptr = tl.load(A_ptr_ptr + g).to(tl.pointer_type(tl.float16))
        B_ptr = tl.load(B_ptr_ptr + g).to(tl.pointer_type(tl.float16))
        C_ptr = tl.load(C_ptr_ptr + g).to(tl.pointer_type(tl.float16))

        A_stride = tl.load(g_stride_ptr + g * 3)
        B_stride = tl.load(g_stride_ptr + g * 3 + 1)
        C_stride = tl.load(g_stride_ptr + g * 3 + 2)
        
        num_tiles_across_M = tl.cdiv(M, BLOCK_M_SIZE)
        num_tiles_across_N = tl.cdiv(N, BLOCK_N_SIZE)

        num_group_tiles = num_tiles_across_M * num_tiles_across_N
    
        while (global_tile_idx < num_last_end + num_group_tiles):
            
            if global_tile_idx >= num_last_end:
                local_tile_id = global_tile_idx - num_last_end
                tile_M_idx = local_tile_id // num_tiles_across_N
                tile_N_idx = local_tile_id % num_tiles_across_N

                offset_am = tile_M_idx * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
                offset_bn = tile_N_idx * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
                offset_k = tl.arange(0, BLOCK_K_SIZE)

                # Initialize pointers for this specific tile
                a_ptrs = A_ptr + (offset_am[:, None] * A_stride) + offset_k[None, :]
                b_ptrs = B_ptr + (offset_k[:, None] * B_stride) + offset_bn[None, :]

                accumulator = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)

                for k in range(0, tl.cdiv(K, BLOCK_K_SIZE)):
                    # Boundary masks
                    mask_A = (offset_am[:, None] < M) & ((k * BLOCK_K_SIZE + offset_k[None, :]) < K)
                    mask_B = ((k * BLOCK_K_SIZE + offset_k[:, None]) < K) & (offset_bn[None, :] < N)

                    A_data = tl.load(a_ptrs, mask=mask_A, other=0.)
                    B_data = tl.load(b_ptrs, mask=mask_B, other=0.)

                    accumulator += tl.dot(A_data, B_data)
                    
                    # Advance pointers to next K-block
                    a_ptrs += BLOCK_K_SIZE
                    b_ptrs += BLOCK_K_SIZE * B_stride

                # Use float32 to match your test bench
                C_out = accumulator.to(tl.float32) 

                mask_C = (offset_am[:, None] < M) & (offset_bn[None, :] < N)
                c_ptrs = C_ptr + C_stride * offset_am[:, None] + offset_bn[None, :]
                tl.store(c_ptrs, C_out, mask=mask_C)

            global_tile_idx += NUM_SM
        num_last_end =  num_last_end + num_group_tiles

def grouped_matmul(group_A, group_B):
    device = group_A[0].device
    A_ptrs = []
    B_ptrs = []
    C_ptrs = []
    group_sizes = []
    group_strides = []
    
    results = []

    for A, B in zip(group_A, group_B):
        M, K = A.shape
        K_verify, N = B.shape
        

        assert K == K_verify
        # Initialize output C
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        results.append(C)

        # Store raw memory addresses (pointers)
        A_ptrs.append(A.data_ptr())
        B_ptrs.append(B.data_ptr())
        C_ptrs.append(C.data_ptr())

        # group_sizes.append((M, N, K))
        # group_strides.append((A.stride(0),B.stride(0),C.stride(0)))

        group_sizes += [(M, N, K)]
        group_strides += [A.stride(0),B.stride(0),C.stride(0)]

    A_ptr_tensor = torch.tensor(A_ptrs, device=device)
    B_ptr_tensor = torch.tensor(B_ptrs, device=device)
    C_ptr_tensor = torch.tensor(C_ptrs, device=device)
    
    g_stride_tensor = torch.tensor(group_strides, device=device, dtype=torch.int32)
    g_size_tensor = torch.tensor(group_sizes, device=device, dtype=torch.int32)

    grid = lambda meta: (
        meta['NUM_SM'],
    )

    grouped_matmul_kernel[grid](
        A_ptr_tensor, B_ptr_tensor, C_ptr_tensor,
        g_size_tensor, g_stride_tensor, g_size_tensor.shape[0]
    )

    return results


def visualize_results(tri_tensor, ref_tensor, group_id):
    # Convert to float32 numpy arrays
    tri = tri_tensor.detach().cpu().to(torch.float32).numpy()
    ref = ref_tensor.detach().cpu().to(torch.float32).numpy()
    diff = np.abs(tri - ref)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Triton Output
    im0 = axes[0].imshow(tri, cmap='viridis', aspect='auto')
    axes[0].set_title(f"Group {group_id}: Triton Output")
    fig.colorbar(im0, ax=axes[0])
    
    # 2. Reference Output
    im1 = axes[1].imshow(ref, cmap='viridis', aspect='auto')
    axes[1].set_title(f"Group {group_id}: Reference")
    fig.colorbar(im1, ax=axes[1])
    
    # 3. Difference (The Error Map)
    im2 = axes[2].imshow(diff, cmap='hot', aspect='auto')
    axes[2].set_title(f"Group {group_id}: Absolute Error")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


def test_GEMM(M_list, N_list, K_list, atol=1e-2, rtol=1e-2):
    print(f"Testing {len(M_list)} groups on {DEVICE}...")
    
    A_group = []
    B_group = []
    
    # Generate data
    for m, n, k in zip(M_list, N_list, K_list):
        A_group.append(torch.randn((m, k), device=DEVICE, dtype=torch.float16))
        B_group.append(torch.randn((k, n), device=DEVICE, dtype=torch.float16))

    # 1. Reference Output
    ref_out = [torch.matmul(a, b) for a, b in zip(A_group, B_group)]

    # 2. Triton Output 
    tri_out = grouped_matmul(A_group, B_group) 

    # 3. Verification
    for i in range(len(M_list)):
        try:
            torch.testing.assert_close(tri_out[i], ref_out[i], atol=atol, rtol=rtol)
            print(f"  ✅ Group {i} ({M_list[i]}x{N_list[i]}): Passed")
        except Exception as e:
            print(f"  ❌ Group {i} ({M_list[i]}x{N_list[i]}): Failed")
            visualize_results(tri_out[i], ref_out[i], i)

            raise e
    
    print("\n⭐ All tests passed!")

if __name__ == "__main__":
    ms = [100, 453, 123, 64]
    ns = [100, 701, 100, 65]
    ks = [20, 85, 233, 32]
    
    test_GEMM(ms, ns, ks)