/**
 * gollek_metal_bridge.m — Objective-C/Metal bridge for Gollek
 * 
 * Optimized for reduced memory pressure and fixed KV cache layout.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ── Globals ───────────────────────────────────────────────────────────────────

static id<MTLDevice>       g_device      = nil;
static id<MTLCommandQueue> g_queue       = nil;
static BOOL                g_initialized = NO;

// Thread-local scratch buffers to avoid allocation in hot loops
static __thread float* tl_k_scratch = NULL;
static __thread float* tl_v_scratch = NULL;
static __thread float* tl_score_scratch = NULL;
static __thread size_t tl_scratch_capacity = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static id<MTLBuffer> wrap_ptr(void* ptr, size_t bytes) {
    return [g_device newBufferWithBytesNoCopy:ptr
                                       length:bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

static void ensure_scratch(size_t required_elements) {
    if (tl_scratch_capacity < required_elements) {
        if (tl_k_scratch) free(tl_k_scratch);
        if (tl_v_scratch) free(tl_v_scratch);
        if (tl_score_scratch) free(tl_score_scratch);
        
        size_t new_cap = required_elements + (required_elements / 4); // +25%
        tl_k_scratch = (float*)malloc(new_cap * sizeof(float));
        tl_v_scratch = (float*)malloc(new_cap * sizeof(float));
        tl_score_scratch = (float*)malloc(new_cap * sizeof(float));
        tl_scratch_capacity = new_cap;
    }
}

// ── Public C API ─────────────────────────────────────────────────────────────

int gollek_metal_init(void) {
    if (g_initialized) return 0;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return -1;
    g_queue = [g_device newCommandQueue];
    g_initialized = YES;
    return 0;
}

long gollek_metal_available_memory(void) {
    if (!g_device) return 0;
    return (long)[g_device recommendedMaxWorkingSetSize];
}

void* gollek_metal_alloc(size_t bytes, size_t align) {
    if (!g_device) return NULL;
    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return buf ? [buf contents] : NULL;
}

int gollek_metal_matmul(void* C, const void* A, const void* B,
                         int M, int K, int N,
                         float alpha, float beta) {
    if (!g_initialized) return -1;
    
    // Fallback to Accelerate for small matrices
    if (M * N < 1024) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)M * K * sizeof(float));
        id<MTLBuffer> bufB = wrap_ptr((void*)B, (size_t)K * N * sizeof(float));

        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication* mmul = [[MPSMatrixMultiplication alloc] initWithDevice:g_device 
            transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:alpha beta:beta];

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_attention(
        void* out,
        const void* Q, const void* K_cache, const void* V_cache,
        const int* block_table, const int* context_lens,
        int B, int T, int H, int D, int block_size, int max_blocks,
        float scale, int is_causal, float soft_cap) {

    if (!g_initialized) return -1;

    for (int b = 0; b < B; b++) {
        int ctx_len = context_lens[b];
        int num_blocks = (ctx_len + block_size - 1) / block_size;
        
        size_t kv_elements = (size_t)ctx_len * H * D;
        ensure_scratch(kv_elements);
        
        float* kPtr = tl_k_scratch;
        float* vPtr = tl_v_scratch;
        const float* kcache = (const float*)K_cache;
        const float* vcache = (const float*)V_cache;

        // Gather from paged layout [phys, H, tokensPerBlock, D]
        for (int blk = 0; blk < num_blocks; blk++) {
            int phys = block_table[b * max_blocks + blk];
            int tokens_in_blk = (blk == num_blocks - 1) ? (ctx_len - blk * block_size) : block_size;
            
            for (int h = 0; h < H; h++) {
                const float* src_k = kcache + ((size_t)phys * H + h) * block_size * D;
                const float* src_v = vcache + ((size_t)phys * H + h) * block_size * D;
                float* dst_k = kPtr + ((size_t)blk * block_size * H + h * block_size) * D;
                float* dst_v = vPtr + ((size_t)blk * block_size * H + h * block_size) * D;
                memcpy(dst_k, src_k, tokens_in_blk * D * sizeof(float));
                memcpy(dst_v, src_v, tokens_in_blk * D * sizeof(float));
            }
        }

        const float* qPtr = (const float*)Q + (size_t)b * T * H * D;
        // scoreBuf needs H * T * ctx_len elements
        ensure_scratch((size_t)H * T * ctx_len > kv_elements ? (size_t)H * T * ctx_len : kv_elements);
        float* scoreBuf = tl_score_scratch;

        for (int h = 0; h < H; h++) {
            for (int t = 0; t < T; t++) {
                const float* qh = qPtr + (t * H + h) * D;
                float* row = scoreBuf + (h * T + t) * ctx_len;
                
                int limit = is_causal ? (t + 1) : ctx_len;

                float mx = -1e30f;
                for (int s = 0; s < limit; s++) {
                    int blk = s / block_size;
                    int tok = s % block_size;
                    const float* kh = kPtr + ((size_t)blk * block_size * H + h * block_size + tok) * D;
                    float dot = 0;
                    for (int d = 0; d < D; d++) dot += qh[d] * kh[d];
                    
                    float score = dot * scale;
                    if (soft_cap > 0.0f) {
                        score = soft_cap * tanhf(score / soft_cap);
                    }
                    row[s] = score;
                    if (row[s] > mx) mx = row[s];
                }

                float sum = 0;
                for (int s = 0; s < limit; s++) { row[s] = expf(row[s] - mx); sum += row[s]; }
                for (int s = 0; s < limit; s++) row[s] /= (sum + 1e-9f);

                // Output = score @ V
                float* oh = (float*)out + (b * T * H + t * H + h) * D;
                memset(oh, 0, D * sizeof(float));
                for (int s = 0; s < limit; s++) {
                    int blk = s / block_size;
                    int tok = s % block_size;
                    const float* vh = vPtr + ((size_t)blk * block_size * H + h * block_size + tok) * D;
                    float weight = row[s];
                    for (int d = 0; d < D; d++) oh[d] += weight * vh[d];
                }
            }
        }
    }
    return 0;
}

int gollek_metal_rmsnorm(void* out, const void* x, const void* weight, int N, float eps, int addOne) {
    const float* xi = (const float*)x;
    const float* wi = (const float*)weight;
    float* oi = (float*)out;
    float ss = 0;
    for (int i = 0; i < N; i++) ss += xi[i] * xi[i];
    float inv = 1.0f / sqrtf(ss / N + eps);
    for (int i = 0; i < N; i++) {
        float w = wi[i];
        if (addOne) w += 1.0f;
        oi[i] = xi[i] * inv * w;
    }
    return 0;
}

int gollek_metal_silu_ffn(void* out, const void* gate, const void* up, int N) {
    const float* g = (const float*)gate;
    const float* u = (const float*)up;
    float* o = (float*)out;
    for (int i = 0; i < N; i++) {
        float gi = g[i];
        o[i] = (gi / (1.0f + expf(-gi))) * u[i];
    }
    return 0;
}

int gollek_metal_device_name(char* buf, int bufSz) {
    if (!g_device) return -1;
    snprintf(buf, bufSz, "%s", [[g_device name] UTF8String]);
    return 0;
}

int gollek_metal_is_unified_memory(void) {
    return (g_device && [g_device hasUnifiedMemory]) ? 1 : 0;
}
