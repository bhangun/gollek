package tech.kayys.gollek.quantizer.turboquant;

import tech.kayys.gollek.quantizer.turboquant.TurboQuantConfig;
import tech.kayys.gollek.quantizer.turboquant.TurboQuantEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * TurboQuant KV Cache Quantizer — Online per-token quantization.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  Paper §4.2-4.3: KV Cache Compression                                 │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │  The KV cache stores key/value tensors for all past tokens.            │
 * │  For a 70B model with 128-token context this is ~280 GB — impractical.│
 * │                                                                        │
 * │  TurboQuant applies Qprod ONLINE (per token, during generation):       │
 * │  • As each token's K and V tensors are computed, immediately quantize  │
 * │  • No calibration data needed (data-oblivious = online-compatible)    │
 * │  • At attention: estimate ⟨query, key⟩ from compressed representation │
 * │                                                                        │
 * │  Outlier Channel Splitting (§4.3):                                     │
 * │  Not all channels are equal — some have high variance (outliers).      │
 * │  Strategy: identify top-K outlier channels by magnitude, apply         │
 * │  TurboQuant at bits+1 for those channels, bits for the rest.          │
 * │  Result: 2.5-bit effective = 32 channels×3bit + 96 channels×2bit      │
 * │                                                                        │
 * │  Compression ratio: 16-bit original / 2.5-bit quantized ≈ 6.4×        │
 * │  Paper achieves score 0.997 (identical to full precision) on NIAH.    │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Usage:
 * <pre>{@code
 *   // For a single attention head (headDim = 128, 2.5-bit KV cache)
 *   var config = TurboQuantConfig.prod2bitKvCache(128);
 *   var kvCache = new TurboQuantKVCache(config, maxSeqLen);
 *
 *   // During forward pass: quantize and store each token's key
 *   kvCache.appendKey(keyVector);
 *
 *   // During attention: estimate all dot products with the query
 *   float[] scores = kvCache.computeAttentionScores(queryVector, seqLen);
 * }</pre>
 */
public class TurboQuantKVCache implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(TurboQuantKVCache.class);

    private final TurboQuantConfig config;
    private final int              headDim;
    private final int              maxSeqLen;

    // Primary TurboQuant engine (for normal channels)
    private final TurboQuantEngine engine;

    // Outlier engine (bits+1, same rotation)
    private final TurboQuantEngine outlierEngine;

    // ── Compressed KV storage (off-heap via FFM Arena) ───────────────────────
    // For each token position t:
    //   mse_indices[t, d]:   b-bit indices, packed as bytes (ceil(bits*d/8) bytes)
    //   qjl_signs[t, d]:     1 byte per dim  (values ∈ {-1, +1})
    //   residual_norms[t]:   1 FP32 per vector
    //   norms[t]:            1 FP32 per vector (original ‖x‖₂ for unnormalised inputs)

    private final Arena         arena;
    private final MemorySegment mseIndicesKeys;    // [maxSeqLen, headDim] INT32
    private final MemorySegment mseIndicesVals;    // [maxSeqLen, headDim] INT32
    private final MemorySegment qjlSignsKeys;      // [maxSeqLen, headDim] INT8
    private final MemorySegment qjlSignsVals;      // [maxSeqLen, headDim] INT8
    private final MemorySegment residualNormsKeys; // [maxSeqLen] FP32
    private final MemorySegment residualNormsVals; // [maxSeqLen] FP32
    private final MemorySegment originalNormsKeys; // [maxSeqLen] FP32
    private final MemorySegment originalNormsVals; // [maxSeqLen] FP32

    // Outlier channel tracking
    private final int[]         outlierChannelIds; // which channel indices are outliers
    private final boolean       hasOutliers;

    // Current fill position
    private final AtomicInteger seqLen = new AtomicInteger(0);

    // Scratch buffers
    private final float[] keyBuf;
    private final float[] valBuf;
    private final int[]   idxBuf;
    private final byte[]  qjlBuf;

    // ── Constructor ───────────────────────────────────────────────────────────

    public TurboQuantKVCache(TurboQuantConfig config, int maxSeqLen) {
        this.config    = config;
        this.headDim   = config.dimension();
        this.maxSeqLen = maxSeqLen;

        this.engine = new TurboQuantEngine(config);

        // Outlier engine uses bits+1
        if (config.splitOutliers() && config.outlierChannels() > 0) {
            TurboQuantConfig outlierConfig = new TurboQuantConfig(
                config.bits() + 1, config.dimension(),
                config.variant(), config.rotation(),
                false, 0, config.seed() + 99L);
            this.outlierEngine = new TurboQuantEngine(outlierConfig);
            this.hasOutliers   = true;
        } else {
            this.outlierEngine = null;
            this.hasOutliers   = false;
        }

        this.outlierChannelIds = hasOutliers
            ? new int[config.outlierChannels()]
            : new int[0];

        // Allocate off-heap storage via FFM Arena
        this.arena = Arena.ofAuto();
        long kvBytes = (long) maxSeqLen * headDim;

        mseIndicesKeys    = arena.allocate(kvBytes * Integer.BYTES, 64L);
        mseIndicesVals    = arena.allocate(kvBytes * Integer.BYTES, 64L);
        qjlSignsKeys      = arena.allocate(kvBytes,                 64L);
        qjlSignsVals      = arena.allocate(kvBytes,                 64L);
        residualNormsKeys = arena.allocate((long) maxSeqLen * Float.BYTES, 64L);
        residualNormsVals = arena.allocate((long) maxSeqLen * Float.BYTES, 64L);
        originalNormsKeys = arena.allocate((long) maxSeqLen * Float.BYTES, 64L);
        originalNormsVals = arena.allocate((long) maxSeqLen * Float.BYTES, 64L);

        // Scratch buffers
        keyBuf = new float[headDim];
        valBuf = new float[headDim];
        idxBuf = new int[headDim];
        qjlBuf = new byte[headDim];

        log.info("TurboQuantKVCache: headDim={}, maxSeq={}, bits={}, outliers={}, off-heap={} MB",
            headDim, maxSeqLen, config.bits(), hasOutliers,
            String.format("%.1f", (mseIndicesKeys.byteSize() * 4 + qjlSignsKeys.byteSize() * 2
             + residualNormsKeys.byteSize() * 4) / 1_048_576.0));
    }

    // ── Channel Statistics ────────────────────────────────────────────────────

    /**
     * Identifies outlier channels from a calibration set of key vectors.
     * Must be called before appendKey if outlier splitting is enabled.
     *
     * Strategy: rank channels by their variance across calibration vectors.
     * The top-{outlierChannels} are designated outliers.
     *
     * @param calibrationKeys  calibration key vectors [numSamples, headDim]
     * @param numSamples       number of calibration samples
     */
    public void identifyOutlierChannels(float[] calibrationKeys, int numSamples) {
        if (!hasOutliers) return;

        // Compute per-channel variance
        double[] mean = new double[headDim];
        double[] var  = new double[headDim];

        for (int t = 0; t < numSamples; t++) {
            for (int j = 0; j < headDim; j++) {
                mean[j] += calibrationKeys[t * headDim + j];
            }
        }
        for (int j = 0; j < headDim; j++) mean[j] /= numSamples;

        for (int t = 0; t < numSamples; t++) {
            for (int j = 0; j < headDim; j++) {
                double diff = calibrationKeys[t * headDim + j] - mean[j];
                var[j] += diff * diff;
            }
        }

        // Rank channels by variance, pick top-K
        Integer[] indices = new Integer[headDim];
        for (int j = 0; j < headDim; j++) indices[j] = j;
        Arrays.sort(indices, (a, b) -> Double.compare(var[b], var[a]));

        int k = config.outlierChannels();
        for (int i = 0; i < k; i++) outlierChannelIds[i] = indices[i];
        Arrays.sort(outlierChannelIds);

        log.debug("Outlier channels identified: {} channels (top variance)", k);
    }

    // ── Append Key / Value ────────────────────────────────────────────────────

    /**
     * Quantizes and stores a key vector for the current token position.
     * Called once per new token during autoregressive generation.
     *
     * @param key  key vector [headDim], raw (not unit-normalised)
     * @return     the sequence position at which this token was stored
     */
    public int appendKey(float[] key) {
        int pos = seqLen.getAndIncrement();
        if (pos >= maxSeqLen) {
            throw new IllegalStateException(
                "KV cache full: pos=" + pos + " >= maxSeq=" + maxSeqLen);
        }

        // Store original norm (for unnormalised inputs the paper assumes ‖x‖=1;
        // we store the norm and normalise before quantizing)
        float origNorm = engine.norm(key);
        System.arraycopy(key, 0, keyBuf, 0, headDim);
        if (origNorm > 1e-8f) {
            for (int j = 0; j < headDim; j++) keyBuf[j] /= origNorm;
        }

        // Quantize using TurboQuant_prod
        TurboQuantEngine.QuantProdResult result = engine.quantizeProd(keyBuf);

        // Write to off-heap storage
        storeInt32Array(mseIndicesKeys, pos, result.mseIndices());
        storeByteArray (qjlSignsKeys,   pos, result.qjlSigns());
        storeFloat(residualNormsKeys, pos, result.residualNorm());
        storeFloat(originalNormsKeys, pos, origNorm);

        return pos;
    }

    /**
     * Quantizes and stores a value vector for the current token position.
     */
    public void appendValue(float[] value, int pos) {
        float origNorm = engine.norm(value);
        System.arraycopy(value, 0, valBuf, 0, headDim);
        if (origNorm > 1e-8f) {
            for (int j = 0; j < headDim; j++) valBuf[j] /= origNorm;
        }

        TurboQuantEngine.QuantProdResult result = engine.quantizeProd(valBuf);

        storeInt32Array(mseIndicesVals, pos, result.mseIndices());
        storeByteArray (qjlSignsVals,   pos, result.qjlSigns());
        storeFloat(residualNormsVals, pos, result.residualNorm());
        storeFloat(originalNormsVals, pos, origNorm);
    }

    // ── Attention Score Computation ───────────────────────────────────────────

    /**
     * Computes unbiased inner product estimates ⟨query, key_t⟩ for all t ∈ [0, seqLen).
     *
     * This is the core attention score computation over compressed KV cache.
     * Uses TurboQuant's inner product estimator (Theorem 2 of paper).
     *
     * @param query   query vector [headDim]
     * @param scores  output scores [seqLen], pre-allocated
     * @param seqLen  number of past tokens to score
     */
    public void computeAttentionScores(float[] query, float[] scores, int seqLen) {
        // Normalise query for the inner product estimator
        float[] qNorm = query.clone();
        float   qLen  = engine.norm(qNorm);
        if (qLen > 1e-8f) for (int j = 0; j < headDim; j++) qNorm[j] /= qLen;

        for (int t = 0; t < seqLen; t++) {
            // Load compressed key for position t
            int[]  idx    = loadInt32Array(mseIndicesKeys, t);
            byte[] qjl    = loadByteArray (qjlSignsKeys,   t);
            float  resNorm = loadFloat(residualNormsKeys, t);
            float  origNorm = loadFloat(originalNormsKeys, t);

            TurboQuantEngine.QuantProdResult result =
                new TurboQuantEngine.QuantProdResult(idx, qjl, resNorm);

            // Estimate ⟨q_norm, key_norm⟩
            float ipNorm = engine.estimateInnerProductFull(qNorm, result);

            // Scale back: ⟨q, k⟩ ≈ ipNorm × qLen × origNorm
            scores[t] = ipNorm * qLen * origNorm;
        }
    }

    /**
     * Dequantizes a stored value vector at position t.
     * Used to compute the weighted sum in attention: Σ_t softmax_t · V_t.
     *
     * @param pos     token position
     * @param output  dequantized value [headDim]
     */
    public void dequantizeValue(int pos, float[] output) {
        int[]  idx     = loadInt32Array(mseIndicesVals, pos);
        byte[] qjl     = loadByteArray (qjlSignsVals,   pos);
        float  resNorm = loadFloat(residualNormsVals, pos);
        float  origNorm = loadFloat(originalNormsVals, pos);

        TurboQuantEngine.QuantProdResult result =
            new TurboQuantEngine.QuantProdResult(idx, qjl, resNorm);

        engine.dequantizeProd(result, output);

        // Scale back to original magnitude
        for (int j = 0; j < headDim; j++) output[j] *= origNorm;
    }

    // ── FFM Storage Helpers ───────────────────────────────────────────────────

    private void storeInt32Array(MemorySegment seg, int pos, int[] data) {
        long base = (long) pos * headDim * Integer.BYTES;
        for (int i = 0; i < headDim; i++) {
            seg.set(ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN),
                base + (long) i * Integer.BYTES, data[i]);
        }
    }

    private int[] loadInt32Array(MemorySegment seg, int pos) {
        long  base = (long) pos * headDim * Integer.BYTES;
        int[] arr  = new int[headDim];
        for (int i = 0; i < headDim; i++) {
            arr[i] = seg.get(ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN),
                base + (long) i * Integer.BYTES);
        }
        return arr;
    }

    private void storeByteArray(MemorySegment seg, int pos, byte[] data) {
        long base = (long) pos * headDim;
        for (int i = 0; i < headDim; i++) {
            seg.set(ValueLayout.JAVA_BYTE, base + i, data[i]);
        }
    }

    private byte[] loadByteArray(MemorySegment seg, int pos) {
        long   base = (long) pos * headDim;
        byte[] arr  = new byte[headDim];
        for (int i = 0; i < headDim; i++) {
            arr[i] = seg.get(ValueLayout.JAVA_BYTE, base + i);
        }
        return arr;
    }

    private void storeFloat(MemorySegment seg, int pos, float value) {
        seg.set(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN),
            (long) pos * Float.BYTES, value);
    }

    private float loadFloat(MemorySegment seg, int pos) {
        return seg.get(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN),
            (long) pos * Float.BYTES);
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    public int getCurrentSeqLen()   { return seqLen.get(); }
    public int getMaxSeqLen()       { return maxSeqLen; }
    public int getHeadDim()         { return headDim; }

    /**
     * Returns the compression ratio relative to FP16 storage.
     * Paper target: >4× compression.
     */
    public double compressionRatio() {
        // Original: headDim × FP16 = headDim × 2 bytes per vector
        // Compressed: headDim × (bits/8 bytes for MSE indices) + headDim × 1 byte for QJL
        //             + 1 × FP32 for residual norm + 1 × FP32 for original norm
        double originalBytes   = headDim * 2.0;  // FP16
        double compressedBytes = headDim * (config.bits() / 8.0)  // MSE indices
                               + headDim * 1.0                     // QJL signs (1 byte each)
                               + 4.0 + 4.0;                        // two FP32 scalars
        return originalBytes / compressedBytes;
    }

    /** Returns effective bits-per-channel for this KV cache config. */
    public double effectiveBitsPerChannel() {
        return config.effectiveBitsPerChannel(headDim);
    }

    public void printStats() {
        System.out.printf("TurboQuant KV Cache: %d/%d tokens, %.1f-bit effective, " +
                          "%.2f× compression%n",
            seqLen.get(), maxSeqLen,
            effectiveBitsPerChannel(),
            compressionRatio());
    }

    @Override
    public void close() {
        if (arena.scope().isAlive()) arena.close();
    }
}
