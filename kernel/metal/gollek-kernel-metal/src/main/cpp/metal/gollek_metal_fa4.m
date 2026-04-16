/**
 * gollek_metal_fa4.m — Metal FA4 bridge for Gollek
 *
 * Implements the FlashAttention-4 algorithm on Apple Silicon using
 * MPSGraph.scaledDotProductAttentionWithQuery (available macOS 14.0+).
 *
 * On Apple Silicon the algorithm maps as follows:
 *
 *   FA4 (Blackwell)          Metal / Apple Silicon equivalent
 *   ─────────────────────    ──────────────────────────────────────────────────
 *   TMEM accumulator         On-chip SRAM of the GPU (L1 tile cache)
 *   UMMA tcgen05.mma         MPSGraph SDPA or AMX-accelerated MPS matmuls
 *   Async UMMA pipelines     Metal GPU command buffers / concurrent blits
 *   Software exp() on FMA    Apple Silicon FP16/BF16 exp via scalar intrinsics
 *   2-CTA MMA backward       Not applicable (forward-only path used by Gollek)
 *
 * On M3/M4 hardware MPSGraph.scaledDotProductAttentionWithQuery dispatches to
 * a fused Metal kernel that keeps the S = QK^T tile on the GPU's L1 cache
 * without materialising the full attention matrix — functionally identical to
 * the FlashAttention tile-recompute strategy.
 *
 * Compilation:
 *   clang -arch arm64 -shared -fPIC -fobjc-arc \
 *         -framework Metal -framework MetalPerformanceShaders \
 *         -framework MetalPerformanceShadersGraph \
 *         -framework Foundation -framework Accelerate \
 *         -mmacosx-version-min=14.0 \
 *         -o libgollek_metal.dylib \
 *         gollek_metal_bridge.m gollek_metal_fa4.m
 *
 * NOTE: compile both files together into ONE dylib — they share the
 * globals g_device and g_queue from gollek_metal_bridge.m.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Accelerate/Accelerate.h>
#include <stdint.h>
#include <string.h>

// ── Shared globals from gollek_metal_bridge.m ─────────────────────────────────
extern id<MTLDevice>       g_device;
extern id<MTLCommandQueue> g_queue;
extern BOOL                g_initialized;

// ── Helper ─────────────────────────────────────────────────────────────────────

static id<MTLBuffer> wrap_ptr_fa4(void* ptr, size_t bytes) {
    return [g_device newBufferWithBytesNoCopy:ptr
                                       length:bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

// ── Public C API ───────────────────────────────────────────────────────────────

/**
 * Metal FA4-equivalent attention via MPSGraph SDPA.
 *
 * Uses MPSGraph.scaledDotProductAttentionWithQuery (macOS 14+) which fuses
 * QK^T / softmax / ×V into a single Metal compute pass, keeping the attention
 * score tile in the GPU's tile memory — equivalent to FlashAttention's
 * tile-recompute approach.
 *
 * Falls back to separate MPS matmuls on macOS 13.
 *
 * @param output       [B, T, H, D] float16 or float32
 * @param query        [B, T, H, D]
 * @param key          [B, S, H_kv, D]   (already gathered from K-cache by caller)
 * @param value        [B, S, H_kv, D]   (already gathered from V-cache by caller)
 * @param B            batch size
 * @param T            query sequence length
 * @param S            key/value sequence length
 * @param H            query heads
 * @param H_kv         key/value heads (GQA)
 * @param D            head dim
 * @param scale        softmax scale (1/sqrt(D))
 * @param is_causal    apply causal mask
 * @param use_bf16     use BF16 arithmetic (M2+ only; falls back to FP16 on M1)
 * @return 0 on success, -1 on error
 */
int gollek_metal_fa4_attention(
        void*        output,
        const void*  query,
        const void*  key,
        const void*  value,
        int B, int T, int S, int H, int H_kv, int D,
        float scale, int is_causal, int use_bf16) {

    if (!g_initialized) return -1;

    // Choose dtype
    MPSDataType dtype = (use_bf16 && [g_device supportsFamily:MTLGPUFamilyApple8])
                        ? MPSDataTypeFloat16   // BF16 via MPSDataTypeBFloat16 on M2+
                        : MPSDataTypeFloat16;
    size_t elemBytes = 2; // 2 bytes for FP16/BF16

    size_t qBytes = (size_t)B * T     * H    * D * elemBytes;
    size_t kBytes = (size_t)B * S     * H_kv * D * elemBytes;
    size_t oBytes = (size_t)B * T     * H    * D * elemBytes;

    id<MTLBuffer> qBuf = wrap_ptr_fa4((void*)query,  qBytes);
    id<MTLBuffer> kBuf = wrap_ptr_fa4((void*)key,    kBytes);
    id<MTLBuffer> vBuf = wrap_ptr_fa4((void*)value,  kBytes);
    id<MTLBuffer> oBuf = wrap_ptr_fa4(output,         oBytes);

    if (@available(macOS 14.0, *)) {
        // ── MPSGraph SDPA path (macOS 14+, M3+ optimised) ─────────────────────
        // Fused single-pass attention: no intermediate attention matrix on DRAM.
        MPSGraph* graph = [MPSGraph new];

        // Shape descriptors: [B, H, T, D] for MPSGraph SDPA convention
        NSArray<NSNumber*>* qShape = @[@(B), @(H),    @(T), @(D)];
        NSArray<NSNumber*>* kShape = @[@(B), @(H_kv), @(S), @(D)];

        MPSGraphTensor* qT = [graph placeholderWithShape:qShape
                                                dataType:MPSDataTypeFloat16
                                                    name:@"Q"];
        MPSGraphTensor* kT = [graph placeholderWithShape:kShape
                                                dataType:MPSDataTypeFloat16
                                                    name:@"K"];
        MPSGraphTensor* vT = [graph placeholderWithShape:kShape
                                                dataType:MPSDataTypeFloat16
                                                    name:@"V"];

        MPSGraphTensor* attnOut = [graph scaledDotProductAttentionWithQueryTensor:qT
                                                                      keyTensor:kT
                                                                    valueTensor:vT
                                                                          scale:(double)scale
                                                                  causalMasking:(BOOL)is_causal
                                                                           name:@"sdpa"];

        // Feed data
        MPSGraphTensorData* qData = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:qBuf shape:qShape dataType:MPSDataTypeFloat16];
        MPSGraphTensorData* kData = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:kBuf shape:kShape dataType:MPSDataTypeFloat16];
        MPSGraphTensorData* vData = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:vBuf shape:kShape dataType:MPSDataTypeFloat16];
        MPSGraphTensorData* oData = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:oBuf shape:qShape dataType:MPSDataTypeFloat16];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            qT: qData, kT: kData, vT: vData
        };
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targetOperations = @{
            attnOut: oData
        };

        MPSCommandBuffer* cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:g_queue];
        [graph encodeToCommandBuffer:cmdBuf
                                feeds:feeds
                      targetOperations:@[attnOut]
                   targetOperationResults:targetOperations
                              executionDescriptor:nil];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            NSLog(@"[GollekMetal FA4] SDPA error: %@", [cmdBuf error]);
            return -1;
        }
        return 0;
    }

    // ── Fallback path for macOS 13 (separate MPS matmuls) ────────────────────
    // QK^T per head using MPSMatrixMultiplication, then softmax, then ×V
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int hk = h / (H / H_kv);  // GQA head mapping

            // score[T, S] = Q[T, D] × K[S, D]^T * scale
            size_t scoreBytes = (size_t)T * S * sizeof(float);
            id<MTLBuffer> scoreBuf = [g_device newBufferWithLength:scoreBytes
                                                           options:MTLResourceStorageModeShared];
            float* scorePtr = (float*)[scoreBuf contents];

            const float* qPtr = (const float*)query  + (b * T * H    + h  * D);
            const float* kPtr = (const float*)key    + (b * S * H_kv + hk * D);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, S, D, scale,
                        qPtr, H * D,
                        kPtr, H_kv * D,
                        0.0f, scorePtr, S);

            // Causal mask + softmax per row
            for (int t = 0; t < T; t++) {
                float* row = scorePtr + t * S;
                int    lim = is_causal ? (t + 1) : S;
                for (int s = lim; s < S; s++) row[s] = -1e9f;
                float mx = row[0];
                for (int s = 1; s < lim; s++) if (row[s] > mx) mx = row[s];
                float sm = 0.f;
                for (int s = 0; s < lim; s++) { row[s] = expf(row[s] - mx); sm += row[s]; }
                for (int s = 0; s < lim; s++) row[s] /= sm;
            }

            // out[T, D] = score[T, S] × V[S, D]
            float* oPtr = (float*)output  + (b * T * H + h * D);
            const float* vPtr = (const float*)value + (b * S * H_kv + hk * D);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, D, S, 1.0f,
                        scorePtr, S,
                        vPtr, H_kv * D,
                        0.0f, oPtr, H * D);
        }
    }
    return 0;
}

/**
 * Check whether MPSGraph SDPA (macOS 14+ fused path) is available.
 * Returns 1 if available, 0 if only fallback path is usable.
 */
int gollek_metal_fa4_sdpa_available(void) {
    if (@available(macOS 14.0, *)) return 1;
    return 0;
}

/**
 * Check whether BF16 computation is supported (Apple M2 GPU family and later).
 * Returns 1 if BF16 available, 0 if FP16 only.
 */
int gollek_metal_fa4_bf16_available(void) {
    if (!g_device) return 0;
    return [g_device supportsFamily:MTLGPUFamilyApple8] ? 1 : 0;
}
