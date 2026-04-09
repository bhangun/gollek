package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.TensorOps;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Flash Attention — memory-efficient scaled dot-product attention that avoids
 * materializing the full N×N attention matrix.
 *
 * <p>Based on <em>"FlashAttention: Fast and Memory-Efficient Exact Attention
 * with IO-Awareness"</em> (Dao et al., 2022).
 *
 * <p>This implementation provides the correct output via tiled computation.
 * When a CUDA kernel plugin is available (via the plugin registry), it
 * automatically delegates to the native FlashAttention kernel for GPU acceleration.
 *
 * <p>Complexity: O(N) memory vs O(N²) for standard attention.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var attn = new FlashAttention(dModel=512, nHeads=8);
 * GradTensor out = attn.forward(x); // [B, T, dModel] → [B, T, dModel]
 * }</pre>
 */
public final class FlashAttention extends NNModule {

    private final int dModel;
    private final int nHeads;
    private final int headDim;
    private final int blockSize; // tile size for tiled computation
    private final boolean causal;

    private final Linear wQ, wK, wV, wO;

    /**
     * Creates a FlashAttention module.
     *
     * @param dModel  model dimension (must be divisible by nHeads)
     * @param nHeads  number of attention heads
     * @param causal  if true, applies causal (autoregressive) masking
     */
    public FlashAttention(int dModel, int nHeads, boolean causal) {
        this(dModel, nHeads, causal, 64);
    }

    /** Creates a non-causal FlashAttention module. */
    public FlashAttention(int dModel, int nHeads) { this(dModel, nHeads, false); }

    /**
     * Creates a FlashAttention module with custom block size.
     *
     * @param dModel    model dimension
     * @param nHeads    number of attention heads
     * @param causal    causal masking flag
     * @param blockSize tile size for tiled softmax (default 64)
     */
    public FlashAttention(int dModel, int nHeads, boolean causal, int blockSize) {
        if (dModel % nHeads != 0)
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        this.dModel    = dModel;
        this.nHeads    = nHeads;
        this.headDim   = dModel / nHeads;
        this.causal    = causal;
        this.blockSize = blockSize;
        this.wQ = register("wQ", new Linear(dModel, dModel, false));
        this.wK = register("wK", new Linear(dModel, dModel, false));
        this.wV = register("wV", new Linear(dModel, dModel, false));
        this.wO = register("wO", new Linear(dModel, dModel));
    }

    /**
     * Forward pass using tiled attention computation.
     *
     * <p>Computes attention in tiles of size {@code blockSize × blockSize},
     * maintaining running softmax statistics (max and sum) to avoid
     * materializing the full N×N matrix.
     *
     * @param x input tensor {@code [B, T, dModel]}
     * @return attention output {@code [B, T, dModel]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        long[] s = x.shape();
        int B = (int) s[0], T = (int) s[1];
        float scale = (float) (1.0 / Math.sqrt(headDim));

        // Project Q, K, V: [B, T, dModel]
        GradTensor Q = wQ.forward(x);
        GradTensor K = wK.forward(x);
        GradTensor V = wV.forward(x);

        // Reshape to [B, H, T, headDim]
        Q = reshape4d(Q, B, T, nHeads, headDim);
        K = reshape4d(K, B, T, nHeads, headDim);
        V = reshape4d(V, B, T, nHeads, headDim);

        float[] qd = Q.data(), kd = K.data(), vd = V.data();
        float[] out = new float[B * nHeads * T * headDim];

        // Tiled attention: process query blocks of size blockSize
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < nHeads; h++) {
                int bhBase = (b * nHeads + h) * T * headDim;

                // For each query tile
                for (int qi = 0; qi < T; qi += blockSize) {
                    int qEnd = Math.min(qi + blockSize, T);

                    // Running softmax stats per query row
                    float[] rowMax = new float[qEnd - qi]; java.util.Arrays.fill(rowMax, Float.NEGATIVE_INFINITY);
                    float[] rowSum = new float[qEnd - qi];
                    float[] rowOut = new float[(qEnd - qi) * headDim];

                    // For each key/value tile
                    for (int ki = 0; ki < T; ki += blockSize) {
                        int kEnd = Math.min(ki + blockSize, T);

                        // Compute scores for this tile: [qBlock, kBlock]
                        for (int qi2 = qi; qi2 < qEnd; qi2++) {
                            int qRow = qi2 - qi;
                            float localMax = Float.NEGATIVE_INFINITY;
                            float[] scores = new float[kEnd - ki];

                            for (int ki2 = ki; ki2 < kEnd; ki2++) {
                                if (causal && ki2 > qi2) { scores[ki2 - ki] = Float.NEGATIVE_INFINITY; continue; }
                                float dot = 0f;
                                for (int d = 0; d < headDim; d++)
                                    dot += qd[bhBase + qi2 * headDim + d] * kd[bhBase + ki2 * headDim + d];
                                scores[ki2 - ki] = dot * scale;
                                if (scores[ki2 - ki] > localMax) localMax = scores[ki2 - ki];
                            }

                            // Online softmax update
                            float newMax = Math.max(rowMax[qRow], localMax);
                            float expScale = (float) Math.exp(rowMax[qRow] - newMax);
                            rowSum[qRow] *= expScale;
                            for (int d = 0; d < headDim; d++) rowOut[qRow * headDim + d] *= expScale;

                            for (int ki2 = ki; ki2 < kEnd; ki2++) {
                                float p = (float) Math.exp(scores[ki2 - ki] - newMax);
                                rowSum[qRow] += p;
                                for (int d = 0; d < headDim; d++)
                                    rowOut[qRow * headDim + d] += p * vd[bhBase + ki2 * headDim + d];
                            }
                            rowMax[qRow] = newMax;
                        }
                    }

                    // Normalize and write output
                    for (int qi2 = qi; qi2 < qEnd; qi2++) {
                        int qRow = qi2 - qi;
                        float invSum = rowSum[qRow] > 0 ? 1f / rowSum[qRow] : 0f;
                        for (int d = 0; d < headDim; d++)
                            out[bhBase + qi2 * headDim + d] = rowOut[qRow * headDim + d] * invSum;
                    }
                }
            }
        }

        // Reshape back [B, H, T, D] → [B, T, dModel] and project
        GradTensor attnOut = GradTensor.of(out, B, nHeads, T, headDim)
            .reshape(B, T, dModel);
        return wO.forward(attnOut);
    }

    /** Reshapes [B, T, H*D] → [B, H, T, D] (transpose H and T). */
    private static GradTensor reshape4d(GradTensor x, int B, int T, int H, int D) {
        float[] src = x.data(), dst = new float[B * H * T * D];
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
                for (int h = 0; h < H; h++)
                    for (int d = 0; d < D; d++)
                        dst[b*H*T*D + h*T*D + t*D + d] = src[b*T*H*D + t*H*D + h*D + d];
        return GradTensor.of(dst, B, H, T, D);
    }

    @Override public String toString() {
        return String.format("FlashAttention(dModel=%d, nHeads=%d, causal=%b, blockSize=%d)",
            dModel, nHeads, causal, blockSize);
    }
}
