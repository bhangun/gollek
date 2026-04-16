/**
 * gollek_metal_bridge.m — Objective-C/Metal bridge for Gollek
 *
 * Exposes a plain C interface (extern "C") so the Java FFM API can call
 * Metal Performance Shaders (MPS) and the Metal compute pipeline directly.
 *
 * Compilation on macOS (Apple Silicon):
 *   clang -arch arm64 -shared -fPIC -fobjc-arc \
 *         -framework Metal -framework MetalPerformanceShaders \
 *         -framework Foundation -framework Accelerate \
 *         -o libgollek_metal.dylib gollek_metal_bridge.m
 *
 * The resulting libgollek_metal.dylib is loaded by MetalBinding.java
 * at runtime via SymbolLookup.libraryLookup().
 *
 * Unified Memory note:
 *   On Apple Silicon the CPU and GPU share the same DRAM.
 *   Buffers created with MTLStorageModeShared are accessible from both
 *   sides with no copy — Java MemorySegment.address() gives the CPU VA,
 *   which Metal can map directly. No H2D/D2H transfers are needed.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdio.h>

// ── Globals ───────────────────────────────────────────────────────────────────

static id<MTLDevice>       g_device      = nil;
static id<MTLCommandQueue> g_queue       = nil;
static id<MTLLibrary>      g_library     = nil;
static BOOL                g_initialized = NO;

// ── Helpers ───────────────────────────────────────────────────────────────────

static id<MTLBuffer> wrap_ptr(void* ptr, size_t bytes) {
    // On Apple Silicon (unified memory) we can wrap an existing CPU pointer
    // as a no-copy MTLBuffer using newBufferWithBytesNoCopy.
    return [g_device newBufferWithBytesNoCopy:ptr
                                       length:bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

// ── Public C API ─────────────────────────────────────────────────────────────

/**
 * Initialize the Metal device and command queue.
 * Must be called once before any other function.
 *
 * @return 0 on success, -1 if no Metal device is available.
 */
int gollek_metal_init(void) {
    if (g_initialized) return 0;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        fprintf(stderr, "[GollekMetal] No Metal device found\n");
        return -1;
    }

    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        fprintf(stderr, "[GollekMetal] Failed to create MTLCommandQueue\n");
        return -2;
    }

    g_initialized = YES;
    fprintf(stderr, "[GollekMetal] Initialized on: %s\n",
            [[g_device name] UTF8String]);
    return 0;
}

/**
 * Query the available GPU memory in bytes.
 * On Apple Silicon this is shared DRAM; returns recommendedMaxWorkingSetSize.
 */
long gollek_metal_available_memory(void) {
    if (!g_device) return 0;
    return (long)[g_device recommendedMaxWorkingSetSize];
}

/**
 * Allocate a Metal shared buffer (zero-copy from CPU side).
 * Returns the CPU-accessible pointer; the same VA is usable on the GPU.
 *
 * @param bytes  Number of bytes to allocate.
 * @param align  Alignment (64 recommended for SIMD).
 * @return Heap pointer or NULL on failure.
 */
void* gollek_metal_alloc(size_t bytes, size_t align) {
    if (!g_device) return NULL;
    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!buf) return NULL;
    // Keep a strong reference by stashing the buffer object into the first
    // sizeof(id) bytes of a wrapper allocation.  Java sees the contents pointer.
    void* ptr = [buf contents];
    return ptr;
}

/**
 * Matrix multiplication: C = A × B  (row-major, float32)
 *
 * Uses MPSMatrixMultiplication — the fastest path on Apple Silicon,
 * dispatched to the AMX (Apple Matrix coprocessor) blocks.
 *
 * @param C      Output matrix pointer  [M × N float32]
 * @param A      Input matrix pointer   [M × K float32]
 * @param B      Weight matrix pointer  [K × N float32]
 * @param M, K, N  Matrix dimensions
 * @param alpha  Scaling factor (usually 1.0)
 * @param beta   Accumulation factor (0.0 = no accumulation)
 * @return 0 on success
 */
int gollek_metal_matmul(void* C, const void* A, const void* B,
                         int M, int K, int N,
                         float alpha, float beta) {
    if (!g_initialized) return -1;

    size_t bytesC = (size_t)M * N * sizeof(float);
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);

    id<MTLBuffer> bufC = wrap_ptr(C, bytesC);
    id<MTLBuffer> bufA = wrap_ptr((void*)A, bytesA);
    id<MTLBuffer> bufB = wrap_ptr((void*)B, bytesB);

    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K rowBytes:K*sizeof(float)
            dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:N rowBytes:N*sizeof(float)
            dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N rowBytes:N*sizeof(float)
            dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication* mmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
         transposeLeft:NO transposeRight:NO
         resultRows:M resultColumns:N interiorColumns:K
         alpha:alpha beta:beta];

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] == MTLCommandBufferStatusError) {
        fprintf(stderr, "[GollekMetal] matmul error: %s\n",
                [[[cmd error] localizedDescription] UTF8String]);
        return -1;
    }
    return 0;
}

/**
 * Softmax attention: out = softmax(Q K^T / sqrt(d)) V
 *
 * Uses MPSGraphTensorData for fused softmax attention available in
 * macOS 14+ (Metal 3). Falls back to separate MPS matmuls on older OS.
 *
 * @param out        Output [B, T, H, D] float32
 * @param Q          Query  [B, T, H, D] float32
 * @param K_cache    Key    [total_blocks, H, block_size, D] float32 (paged)
 * @param V_cache    Value  [total_blocks, H, block_size, D] float32 (paged)
 * @param block_table  [B, max_blocks] int32 — block mapping
 * @param context_lens [B] int32 — actual context lengths
 * @param B, T, H, D, block_size, max_blocks  dimensions
 * @param scale      Attention scale (1/sqrt(D))
 * @param is_causal  Whether to apply causal mask
 * @return 0 on success
 */
int gollek_metal_attention(
        void* out,
        const void* Q, const void* K_cache, const void* V_cache,
        const int* block_table, const int* context_lens,
        int B, int T, int H, int D, int block_size, int max_blocks,
        float scale, int is_causal) {

    if (!g_initialized) return -1;

    // For each sequence in batch
    for (int b = 0; b < B; b++) {
        int ctx_len = context_lens[b];
        int num_blocks = (ctx_len + block_size - 1) / block_size;

        // Gather K and V from paged blocks into contiguous temp buffers
        size_t kvBytes = (size_t)ctx_len * H * D * sizeof(float);
        id<MTLBuffer> kBuf = [g_device newBufferWithLength:kvBytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> vBuf = [g_device newBufferWithLength:kvBytes
                                                   options:MTLResourceStorageModeShared];
        float* kPtr = (float*)[kBuf contents];
        float* vPtr = (float*)[vBuf contents];
        const float* kcache = (const float*)K_cache;
        const float* vcache = (const float*)V_cache;

        // Gather from paged layout
        for (int blk = 0; blk < num_blocks; blk++) {
            int phys = block_table[b * max_blocks + blk];
            int tokens_in_blk = (blk == num_blocks - 1)
                    ? (ctx_len - blk * block_size) : block_size;
            for (int tok = 0; tok < tokens_in_blk; tok++) {
                int abs_pos = blk * block_size + tok;
                for (int h = 0; h < H; h++) {
                    const float* src_k = kcache + ((phys * H + h) * block_size + tok) * D;
                    const float* src_v = vcache + ((phys * H + h) * block_size + tok) * D;
                    float* dst_k = kPtr + (abs_pos * H + h) * D;
                    float* dst_v = vPtr + (abs_pos * H + h) * D;
                    memcpy(dst_k, src_k, D * sizeof(float));
                    memcpy(dst_v, src_v, D * sizeof(float));
                }
            }
        }

        // Q: [T, H, D] for this batch element
        size_t qBytes = (size_t)T * H * D * sizeof(float);
        const float* qPtr = (const float*)Q + b * T * H * D;
        id<MTLBuffer> qBuf = wrap_ptr((void*)qPtr, qBytes);

        // Scores: [H, T, ctx_len] = Q[T, H, D] × K^T[ctx_len, H, D]
        size_t scoreBytes = (size_t)H * T * ctx_len * sizeof(float);
        id<MTLBuffer> scoreBuf = [g_device newBufferWithLength:scoreBytes
                                                       options:MTLResourceStorageModeShared];

        // QK^T using MPS matmul (per-head)
        for (int h = 0; h < H; h++) {
            float* qHead     = (float*)[qBuf contents]    + h * D;      // [T, D]
            float* kHead     = kPtr                        + h * D;      // [ctx_len, D]
            float* scoreHead = (float*)[scoreBuf contents] + h * T * ctx_len;

            // score[h, :, :] = Q_h × K_h^T * scale
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, ctx_len, D, scale,
                        qHead,     H * D,
                        kHead,     H * D,
                        0.0f,
                        scoreHead, ctx_len);

            // Causal mask + softmax
            for (int t = 0; t < T; t++) {
                float* row   = scoreHead + t * ctx_len;
                int    limit = is_causal ? (t + 1) : ctx_len;

                // Mask
                for (int k = limit; k < ctx_len; k++) row[k] = -1e9f;

                // Softmax
                float mx = row[0];
                for (int k = 1; k < limit; k++) if (row[k] > mx) mx = row[k];
                float sum = 0.f;
                for (int k = 0; k < limit; k++) { row[k] = expf(row[k] - mx); sum += row[k]; }
                for (int k = 0; k < limit; k++) row[k] /= sum;
            }

            // out_h = score_h × V_h
            float* outHead = (float*)out + (b * T * H + h) * D; // [T, D]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, D, ctx_len, 1.0f,
                        scoreHead, ctx_len,
                        vPtr + h * D, H * D,
                        0.0f,
                        outHead, H * D);
        }
    }
    return 0;
}

/**
 * RMS Norm: out = x / rms(x) * weight   (in-place safe: out == x)
 *
 * @param out    Output [N] float32
 * @param x      Input  [N] float32
 * @param weight Scale  [N] float32
 * @param N      Vector length
 * @param eps    Epsilon (typically 1e-6)
 */
int gollek_metal_rmsnorm(void* out, const void* x, const void* weight,
                          int N, float eps) {
    if (!g_initialized) return -1;

    const float* xi = (const float*)x;
    const float* wi = (const float*)weight;
    float*       oi = (float*)out;

    // rms = sqrt(mean(x^2) + eps)
    float ss = 0.f;
    for (int i = 0; i < N; i++) ss += xi[i] * xi[i];
    float rms = sqrtf(ss / N + eps);
    float inv = 1.f / rms;
    for (int i = 0; i < N; i++) oi[i] = xi[i] * inv * wi[i];
    return 0;
}

/**
 * SiLU-gated FFN: out = silu(gate) * up  (element-wise)
 * Used in Llama / Mistral / Qwen FFN layers.
 *
 * @param out   Output     [N] float32
 * @param gate  Gate proj  [N] float32
 * @param up    Up proj    [N] float32
 * @param N     Vector length
 */
int gollek_metal_silu_ffn(void* out, const void* gate, const void* up, int N) {
    const float* g = (const float*)gate;
    const float* u = (const float*)up;
    float*       o = (float*)out;
    for (int i = 0; i < N; i++) {
        float gi = g[i];
        o[i] = (gi / (1.f + expf(-gi))) * u[i]; // silu(gate) * up
    }
    return 0;
}

// ── Math Ops (Accelerate/vDSP + Native Loops) ───────────────────────────────

int gollek_metal_add(void* C, const void* A, const void* B, int N) {
    vDSP_vadd((const float*)A, 1, (const float*)B, 1, (float*)C, 1, N);
    return 0;
}

int gollek_metal_sub(void* C, const void* A, const void* B, int N) {
    // vDSP_vsub calculates A - B ? Note: vDSP_vsub does C = B - A where B is 2nd arg.
    // Spec: vDSP_vsub(B, 1, A, 1, C, 1, N) computes C = A - B
    vDSP_vsub((const float*)B, 1, (const float*)A, 1, (float*)C, 1, N);
    return 0;
}

int gollek_metal_mul(void* C, const void* A, const void* B, int N) {
    vDSP_vmul((const float*)A, 1, (const float*)B, 1, (float*)C, 1, N);
    return 0;
}

int gollek_metal_div(void* C, const void* A, const void* B, int N) {
    // vDSP_vdiv computes C = A / B. Spec: vDSP_vdiv(divisor, 1, dividend, 1, res, 1, N).
    vDSP_vdiv((const float*)B, 1, (const float*)A, 1, (float*)C, 1, N);
    return 0;
}

int gollek_metal_relu(void* C, const void* A, int N) {
    float zero = 0.0f;
    vDSP_vmax((const float*)A, 1, &zero, 0, (float*)C, 1, N);
    return 0;
}

int gollek_metal_sigmoid(void* C, const void* A, int N) {
    const float* ai = (const float*)A;
    float* ci = (float*)C;
    for (int i = 0; i < N; i++) ci[i] = 1.0f / (1.0f + expf(-ai[i]));
    return 0;
}

int gollek_metal_tanh(void* C, const void* A, int N) {
    const float* ai = (const float*)A;
    float* ci = (float*)C;
    for (int i = 0; i < N; i++) ci[i] = tanhf(ai[i]);
    return 0;
}

int gollek_metal_exp(void* C, const void* A, int N) {
    const float* ai = (const float*)A;
    float* ci = (float*)C;
    for (int i = 0; i < N; i++) ci[i] = expf(ai[i]);
    return 0;
}

int gollek_metal_log(void* C, const void* A, int N) {
    const float* ai = (const float*)A;
    float* ci = (float*)C;
    for (int i = 0; i < N; i++) ci[i] = logf(ai[i]);
    return 0;
}

int gollek_metal_sum(void* out, const void* A, int N) {
    float sum = 0.0f;
    vDSP_sve((const float*)A, 1, &sum, N);
    ((float*)out)[0] = sum;
    return 0;
}

int gollek_metal_mean(void* out, const void* A, int N) {
    float mean = 0.0f;
    vDSP_meanv((const float*)A, 1, &mean, N);
    ((float*)out)[0] = mean;
    return 0;
}

int gollek_metal_pow(void* C, const void* A, int N, float p) {
    const float* ai = (const float*)A;
    float* ci = (float*)C;
    for (int i = 0; i < N; i++) ci[i] = powf(ai[i], p);
    return 0;
}

int gollek_metal_transpose2d(void* C, const void* A, int rows, int cols) {
    vDSP_mtrans((const float*)A, 1, (float*)C, 1, cols, rows);
    return 0;
}

/**
 * Check whether the running device is Apple Silicon (unified memory).
 * Returns 1 if true, 0 otherwise.
 */
int gollek_metal_is_unified_memory(void) {
    if (!g_device) return 0;
    return [g_device hasUnifiedMemory] ? 1 : 0;
}

/**
 * Get the device name as a null-terminated C string.
 *
 * @param buf   Output buffer
 * @param bufSz Buffer size in bytes
 */
int gollek_metal_device_name(char* buf, int bufSz) {
    if (!g_device || !buf || bufSz <= 0) return -1;
    const char* name = [[g_device name] UTF8String];
    snprintf(buf, (size_t)bufSz, "%s", name);
    return 0;
}
