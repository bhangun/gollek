/**
 * gollek_kernels.cu — PagedAttention CUDA kernel for Gollek Inference Engine
 *
 * This kernel implements PagedAttention v1 for efficient KV-Cache lookup
 * using block tables. Called from Java via FFM (Foreign Function & Memory API).
 *
 * Compilation:
 *   make -C src/main/cpp
 *
 * The resulting libgollek_kernels.so is loaded at runtime via FFM SymbolLookup.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
// PagedAttention v1 Kernel
// ============================================================================

/**
 * PagedAttention forward kernel.
 *
 * For each query head, this kernel:
 *   1. Reads the block table to find physical K/V block locations
 *   2. Computes Q·K^T attention scores across all blocks
 *   3. Applies softmax normalization
 *   4. Computes weighted sum of V values
 *
 * @param output       Output tensor [num_seqs, num_heads, head_dim]
 * @param query        Query tensor  [num_seqs, num_heads, head_dim]
 * @param key_cache    K cache pool  [total_blocks, num_heads, block_size, head_dim]
 * @param value_cache  V cache pool  [total_blocks, num_heads, block_size, head_dim]
 * @param block_tables Block mapping  [num_seqs, max_blocks_per_seq]
 * @param context_lens Actual context length per sequence [num_seqs]
 * @param num_heads    Number of attention heads
 * @param head_dim     Dimension per head
 * @param block_size   Tokens per block
 * @param max_context  Maximum context length across batch
 * @param scale        Attention scale factor (1.0 / sqrt(head_dim))
 */
__global__ void paged_attention_v1_kernel(
    float* __restrict__ output,
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int num_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale
) {
    // Grid dimensions:
    //   blockIdx.x = sequence index
    //   blockIdx.y = head index
    //   threadIdx.x = head dimension index (up to head_dim)

    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;

    if (dim_idx >= head_dim) return;

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + block_size - 1) / block_size;

    // Load query value for this (seq, head, dim)
    const int q_offset = (seq_idx * num_heads + head_idx) * head_dim + dim_idx;
    const float q_val = query[q_offset] * scale;

    // --- Phase 1: Compute attention scores ---
    // We use shared memory for partial softmax reduction
    extern __shared__ float shared_mem[];
    float* attn_scores = shared_mem;  // [max_context]

    float max_score = -1e20f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        const int tokens_in_block = min(block_size, context_len - block_idx * block_size);

        for (int tok = 0; tok < tokens_in_block; tok++) {
            const int abs_pos = block_idx * block_size + tok;

            // K cache layout: [total_blocks, num_heads, block_size, head_dim]
            const int k_offset = ((physical_block * num_heads + head_idx) * block_size + tok) * head_dim + dim_idx;
            const float k_val = key_cache[k_offset];

            // Each thread computes partial dot product, need warp reduction
            float partial_dot = q_val * k_val;

            // Warp-level reduction for dot product
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                partial_dot += __shfl_down_sync(0xffffffff, partial_dot, offset);
            }

            if (dim_idx == 0) {
                attn_scores[abs_pos] = partial_dot;
                max_score = fmaxf(max_score, partial_dot);
            }
        }
    }

    __syncthreads();

    // --- Phase 2: Softmax ---
    if (dim_idx == 0) {
        float sum_exp = 0.0f;
        for (int i = 0; i < context_len; i++) {
            attn_scores[i] = expf(attn_scores[i] - max_score);
            sum_exp += attn_scores[i];
        }
        for (int i = 0; i < context_len; i++) {
            attn_scores[i] /= sum_exp;
        }
    }

    __syncthreads();

    // --- Phase 3: Weighted sum of V ---
    float acc = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        const int tokens_in_block = min(block_size, context_len - block_idx * block_size);

        for (int tok = 0; tok < tokens_in_block; tok++) {
            const int abs_pos = block_idx * block_size + tok;

            // V cache layout: same as K cache
            const int v_offset = ((physical_block * num_heads + head_idx) * block_size + tok) * head_dim + dim_idx;
            acc += attn_scores[abs_pos] * value_cache[v_offset];
        }
    }

    // Write output
    output[q_offset] = acc;
}


// ============================================================================
// C Interface (extern "C" for FFM compatibility)
// ============================================================================

extern "C" {

/**
 * Launch the PagedAttention kernel from Java via FFM.
 *
 * All pointer arguments are raw device pointers obtained from
 * MemorySegment.address() on the Java side.
 *
 * @return 0 on success, CUDA error code on failure
 */
int paged_attention_launch(
    float* output,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_tables,
    const int* context_lens,
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale
) {
    // Grid: (num_seqs, num_heads)
    // Block: (head_dim)
    dim3 grid(num_seqs, num_heads);
    dim3 block(head_dim);

    // Shared memory for attention scores: max possible context length
    int max_context = max_blocks_per_seq * block_size;
    size_t shared_mem_size = max_context * sizeof(float);

    paged_attention_v1_kernel<<<grid, block, shared_mem_size>>>(
        output, query, key_cache, value_cache,
        block_tables, context_lens,
        num_heads, head_dim, block_size,
        max_blocks_per_seq, scale
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in paged_attention_launch: %s\n",
                cudaGetErrorString(err));
        return (int)err;
    }

    err = cudaDeviceSynchronize();
    return (int)err;
}

/**
 * Query CUDA device availability.
 * @return number of CUDA devices, or 0 if none available
 */
int gollek_cuda_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count;
}

/**
 * Get the name of the current CUDA device.
 * @param buffer output buffer for device name
 * @param bufferSize size of the output buffer
 * @return 0 on success
 */
int gollek_cuda_device_name(char* buffer, int bufferSize) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    snprintf(buffer, bufferSize, "%s", prop.name);
    return 0;
}

} // extern "C"
