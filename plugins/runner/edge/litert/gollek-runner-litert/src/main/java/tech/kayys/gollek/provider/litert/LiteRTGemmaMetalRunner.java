package tech.kayys.gollek.provider.litert;

import lombok.extern.slf4j.Slf4j;
import tech.kayys.gollek.metal.binding.MetalBinding;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;
import java.util.Map;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Multiplatform Gemma-4 inference engine using Gollek's compute backend.
 *
 * <p>Uses {@link MetalBinding} for hardware-accelerated operations on Apple Silicon
 * and automatically falls back to the CPU implementation on all other platforms.
 * Weights are loaded from the companion {@code .task} file (MediaPipe format)
 * since {@code .litertlm} files use opaque NPU-compiled blobs.
 *
 * <p>Architecture (Gemma-4 E2B):
 * <ul>
 *   <li>35 transformer layers with GQA (8 Q heads : 1 KV head)</li>
 *   <li>hidden_dim=1536, head_dim=256, ffn_dim=6144</li>
 *   <li>INT4 quantized weights with per-channel float32 scales</li>
 *   <li>Pre/post attention + FFN RMSNorm with skip connections</li>
 *   <li>SwiGLU FFN: gate * silu(ff1) → down projection</li>
 * </ul>
 */
@Slf4j
public class LiteRTGemmaMetalRunner implements AutoCloseable {

    private final MetalBinding metal;
    private final Path modelPath;
    private final LiteRTTokenizer tokenizer;

    private final Arena arena = Arena.ofShared();
    private Map<String, MemorySegment> weightSegments;
    private final Map<String, MemorySegment> dequantizedWeights = new java.util.concurrent.ConcurrentHashMap<>();
    private boolean initialized = false;

    // ── Architecture Constants (Gemma-4 E2B) ──────────────────────────────────
    private static final int NUM_LAYERS = 35;
    private static final int HIDDEN_DIM = 1536;
    private static final int NUM_Q_HEADS = 8;
    private static final int NUM_KV_HEADS = 1;
    private static final int HEAD_DIM = 256;
    private static final int Q_DIM = NUM_Q_HEADS * HEAD_DIM;   // 2048
    private static final int KV_DIM = NUM_KV_HEADS * HEAD_DIM; // 256
    private static final int FFN_DIM = 6144;
    private static final int MAX_SEQ_LEN = 2048;

    // TFLite type constants
    private static final int TFLITE_INT8 = 9;
    private static final int TFLITE_INT4 = 17;

    // ── KV Cache ──────────────────────────────────────────────────────────────
    // Per-layer KV cache: [seqLen, numKvHeads, headDim]
    private MemorySegment[] kCaches;
    private MemorySegment[] vCaches;
    private int cacheLen = 0;

    public LiteRTGemmaMetalRunner(Path modelPath, LiteRTTokenizer tokenizer) {
        this.modelPath = modelPath;
        this.tokenizer = tokenizer;
        MetalBinding.initialize();
        this.metal = MetalBinding.getInstance();
    }

    // ── Weight Source Resolution ──────────────────────────────────────────────

    private Path resolveWeightSource() {
        String fileName = modelPath.getFileName().toString();
        if (fileName.endsWith(".task")) return modelPath;

        Path dir = modelPath.getParent();
        if (dir == null) dir = Path.of(".");

        String baseName = fileName.replaceAll("(_qualcomm_[^.]+)?\\.litertlm$", "");
        for (String suffix : new String[]{"-web.task", ".task"}) {
            Path candidate = dir.resolve(baseName + suffix);
            if (Files.exists(candidate)) {
                log.info("Found companion .task weight file: {}", candidate.getFileName());
                return candidate;
            }
        }
        log.warn("No companion .task file found; using {}", fileName);
        return modelPath;
    }

    // ── Initialization ───────────────────────────────────────────────────────

    public void initialize() throws IOException {
        log.info("Initializing Gemma inference engine for: {}", modelPath);

        // 1. Initialize compute backend (Metal or CPU fallback)
        if (metal.init() != 0) {
            throw new RuntimeException("Failed to initialize compute backend");
        }
        String device = metal.deviceName();
        log.info("✓ Compute backend: {} (unified_mem={})", device, metal.isUnifiedMemory());

        // 2. Load weights from .task file
        Path weightSource = resolveWeightSource();
        Map<String, LiteRTContainerParser.WeightEntry> weightEntries =
                LiteRTContainerParser.extractWeightMap(weightSource);

        log.info("Extracted {} weight tensors from {}", weightEntries.size(), weightSource.getFileName());

        this.weightSegments = new java.util.HashMap<>();
        try (FileChannel channel = FileChannel.open(weightSource, StandardOpenOption.READ)) {
            for (var entry : weightEntries.values()) {
                if (entry.size() > 0) {
                    MemorySegment seg = channel.map(FileChannel.MapMode.READ_ONLY,
                            entry.offset(), entry.size(), arena);
                    weightSegments.put(entry.name(), seg);
                }
            }
        }
        log.info("✓ {} weight tensors memory-mapped", weightSegments.size());

        // 3. Allocate KV caches (per-layer)
        kCaches = new MemorySegment[NUM_LAYERS];
        vCaches = new MemorySegment[NUM_LAYERS];
        long kvLayerSize = (long) MAX_SEQ_LEN * KV_DIM * 4; // float32
        for (int l = 0; l < NUM_LAYERS; l++) {
            kCaches[l] = arena.allocate(kvLayerSize, 64);
            vCaches[l] = arena.allocate(kvLayerSize, 64);
        }
        cacheLen = 0;
        log.info("✓ KV-Cache allocated ({} MB per layer, {} layers)",
                kvLayerSize * 2 / (1024 * 1024), NUM_LAYERS);

        this.initialized = true;
    }

    // ── Weight Access Helpers ────────────────────────────────────────────────

    private MemorySegment getWeight(String name) {
        MemorySegment seg = weightSegments.get(name);
        if (seg == null) {
            throw new RuntimeException("Missing weight tensor: " + name);
        }
        return seg;
    }

    private boolean hasWeight(String name) {
        return weightSegments.containsKey(name);
    }

    /**
     * Dequantize INT4 packed weight tensor to float32.
     * INT4 packs 2 values per byte (little-endian nibble order).
     * Each output element = int4_val * scale[output_channel].
     *
     * @param weight INT4 packed data [outDim * inDim / 2 bytes]
     * @param scale  float32 per-output-channel scales [outDim]
     * @param outDim output dimension (rows)
     * @param inDim  input dimension (cols)
     * @return float32 MemorySegment [outDim × inDim]
     */
    private MemorySegment dequantizeInt4(MemorySegment weight, MemorySegment scale,
                                          int outDim, int inDim, Arena a) {
        long floatSize = (long) outDim * inDim * 4;
        MemorySegment out = a.allocate(floatSize, 64);
        
        byte[] wArr = weight.toArray(ValueLayout.JAVA_BYTE);
        float[] sArr = scale.toArray(ValueLayout.JAVA_FLOAT);

        for (int row = 0; row < outDim; row++) {
            float s = sArr[row];
            int rowByteOffset = row * inDim / 2;
            long outRowOffset = (long) row * inDim;

            for (int col = 0; col < inDim; col += 2) {
                byte packed = wArr[rowByteOffset + col / 2];
                int lo = (packed & 0x0F);
                int hi = (packed >> 4) & 0x0F;
                if (lo >= 8) lo -= 16;
                if (hi >= 8) hi -= 16;

                out.setAtIndex(ValueLayout.JAVA_FLOAT, outRowOffset + col, lo * s);
                if (col + 1 < inDim) {
                    out.setAtIndex(ValueLayout.JAVA_FLOAT, outRowOffset + col + 1, hi * s);
                }
            }
        }
        return out;
    }

    /**
     * Dequantize INT8 weight tensor to float32.
     */
    private MemorySegment dequantizeInt8(MemorySegment weight, MemorySegment scale,
                                          int outDim, int inDim, Arena a) {
        long floatSize = (long) outDim * inDim * 4;
        MemorySegment out = a.allocate(floatSize, 64);

        byte[] wArr = weight.toArray(ValueLayout.JAVA_BYTE);
        float[] sArr = scale.toArray(ValueLayout.JAVA_FLOAT);

        for (int row = 0; row < outDim; row++) {
            float s = sArr[row];
            int rowOffset = row * inDim;
            long outOffset = (long) row * inDim;
            for (int col = 0; col < inDim; col++) {
                byte val = wArr[rowOffset + col];
                out.setAtIndex(ValueLayout.JAVA_FLOAT, outOffset + col, val * s);
            }
        }
        return out;
    }

    // ── Core Operations ─────────────────────────────────────────────────────

    /** RMS Normalization: out = x * weight / rms(x) */
    private void rmsNorm(MemorySegment out, MemorySegment x, MemorySegment weight, int dim) {
        metal.rmsNorm(out, x, weight, dim, 1e-6f);
    }

    /** Matrix multiplication with auto-dequantization caching. */
    private void linearForward(MemorySegment out, MemorySegment input,
                                String weightName, int outDim, int inDim, Arena stepArena) {
        String cacheKey = weightName + ".w_float32";
        MemorySegment wFloat = dequantizedWeights.get(cacheKey);

        if (wFloat == null) {
            MemorySegment w = getWeight(weightName + ".w");
            MemorySegment scale = getWeight(weightName + ".w_quantized_scale");

            // Dequantize based on type - use main arena for persistent caching
            long expectedInt4Size = (long) outDim * inDim / 2;
            long expectedInt8Size = (long) outDim * inDim;

            if (w.byteSize() == expectedInt4Size) {
                wFloat = dequantizeInt4(w, scale, outDim, inDim, arena);
            } else if (w.byteSize() == expectedInt8Size) {
                wFloat = dequantizeInt8(w, scale, outDim, inDim, arena);
            } else {
                wFloat = w; // Assume float32
            }
            dequantizedWeights.put(cacheKey, wFloat);
        }

        // out = input[1, inDim] × wFloat^T[inDim, outDim] → [1, outDim]
        metal.matmul(out, input, wFloat, 1, inDim, outDim, 1.0f, 0.0f);
    }

    /** Apply RoPE (Rotary Position Embedding) to Q or K. */
    private void applyRope(MemorySegment qk, int dim, int pos) {
        // RoPE: for each pair (i, i+1) apply rotation by θ_i * pos
        // θ_i = 1 / 10000^(2i/dim)
        for (int i = 0; i < dim; i += 2) {
            float freq = (float) (1.0 / Math.pow(10000.0, (double) i / dim));
            float angle = freq * pos;
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);

            float r0 = qk.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float r1 = qk.getAtIndex(ValueLayout.JAVA_FLOAT, i + 1);

            qk.setAtIndex(ValueLayout.JAVA_FLOAT, i, r0 * cos - r1 * sin);
            qk.setAtIndex(ValueLayout.JAVA_FLOAT, i + 1, r0 * sin + r1 * cos);
        }
    }

    /** Softmax in-place over logits[0..size-1]. */
    private void softmax(MemorySegment logits, int size) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            float v = logits.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (v > max) max = v;
        }
        float sum = 0;
        for (int i = 0; i < size; i++) {
            float v = (float) Math.exp(logits.getAtIndex(ValueLayout.JAVA_FLOAT, i) - max);
            logits.setAtIndex(ValueLayout.JAVA_FLOAT, i, v);
            sum += v;
        }
        for (int i = 0; i < size; i++) {
            logits.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                    logits.getAtIndex(ValueLayout.JAVA_FLOAT, i) / sum);
        }
    }

    /** Argmax over float array. */
    private int argmax(MemorySegment logits, int size) {
        int best = 0;
        float bestVal = logits.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        for (int i = 1; i < size; i++) {
            float v = logits.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    // ── Transformer Forward Pass ────────────────────────────────────────────

    /**
     * Single decode step through the full transformer.
     *
     * @param tokenId input token
     * @param pos     position in the sequence
     * @param a       arena for temporary allocations
     * @return logits over vocabulary
     */
    private MemorySegment forwardStep(int tokenId, int pos, Arena a) {
        // ── 1. EMBEDDING ──
        // input_embedding.w is INT8: [vocabSize, hiddenDim]
        MemorySegment embW = getWeight("transformer.embedder.input_embedding.w");
        MemorySegment embScale = getWeight("transformer.embedder.input_embedding.w_quantized_scale");

        MemorySegment x = a.allocate((long) HIDDEN_DIM * 4, 64);
        // Dequantize single row for the token
        float rowScale = embScale.getAtIndex(ValueLayout.JAVA_FLOAT, tokenId);
        long rowOffset = (long) tokenId * HIDDEN_DIM;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            byte val = embW.get(ValueLayout.JAVA_BYTE, rowOffset + i);
            x.setAtIndex(ValueLayout.JAVA_FLOAT, i, val * rowScale);
        }

        MemorySegment residual = a.allocate((long) HIDDEN_DIM * 4, 64);
        MemorySegment normed = a.allocate((long) HIDDEN_DIM * 4, 64);

        // Temporary buffers for projections
        MemorySegment q = a.allocate((long) Q_DIM * 4, 64);
        MemorySegment k = a.allocate((long) KV_DIM * 4, 64);
        MemorySegment v = a.allocate((long) KV_DIM * 4, 64);
        MemorySegment attnOut = a.allocate((long) Q_DIM * 4, 64);
        MemorySegment projOut = a.allocate((long) HIDDEN_DIM * 4, 64);
        MemorySegment gate = a.allocate((long) FFN_DIM * 4, 64);
        MemorySegment up = a.allocate((long) FFN_DIM * 4, 64);
        MemorySegment ffnOut = a.allocate((long) FFN_DIM * 4, 64);
        MemorySegment down = a.allocate((long) HIDDEN_DIM * 4, 64);

        // ── 2. TRANSFORMER LAYERS ──
        for (int l = 0; l < NUM_LAYERS; l++) {
            String prefix = "transformer.layer_" + l + ".";

            // ── Pre-attention RMSNorm ──
            rmsNorm(normed, x, getWeight(prefix + "pre_attention_norm.scale"), HIDDEN_DIM);
            // Save residual
            MemorySegment.copy(x, 0, residual, 0, (long) HIDDEN_DIM * 4);

            // ── Q/K/V Projections ──
            linearForward(q, normed, prefix + "attn.q", Q_DIM, HIDDEN_DIM, a);

            boolean hasKV = hasWeight(prefix + "attn.k.w");
            if (hasKV) {
                linearForward(k, normed, prefix + "attn.k", KV_DIM, HIDDEN_DIM, a);
                linearForward(v, normed, prefix + "attn.v", KV_DIM, HIDDEN_DIM, a);

                if (hasWeight(prefix + "attn.k_norm.scale")) {
                    MemorySegment kNormW = getWeight(prefix + "attn.k_norm.scale");
                    rmsNorm(k, k, kNormW, KV_DIM);
                }

                // Apply RoPE to K
                applyRope(k, KV_DIM, pos);

                // ── KV Cache Update ──
                MemorySegment.copy(k, 0, kCaches[l], (long) pos * KV_DIM * 4, (long) KV_DIM * 4);
                MemorySegment.copy(v, 0, vCaches[l], (long) pos * KV_DIM * 4, (long) KV_DIM * 4);
            }

            // ── Q Norm (Gemma-4 uses per-head normalization) ──
            if (hasWeight(prefix + "attn.q_norm.scale")) {
                MemorySegment qNormW = getWeight(prefix + "attn.q_norm.scale");
                // Apply per-head norm: Q is [numQHeads * headDim], norm is [headDim]
                for (int h = 0; h < NUM_Q_HEADS; h++) {
                    long headOff = (long) h * HEAD_DIM * 4;
                    rmsNorm(q.asSlice(headOff, (long) HEAD_DIM * 4),
                            q.asSlice(headOff, (long) HEAD_DIM * 4), qNormW, HEAD_DIM);
                }
            }

            // ── RoPE on Q ──
            for (int h = 0; h < NUM_Q_HEADS; h++) {
                applyRope(q.asSlice((long) h * HEAD_DIM * 4, (long) HEAD_DIM * 4), HEAD_DIM, pos);
            }

            // Find the active KV cache for this layer (for CLA/shared KV)
            MemorySegment activeKCache = kCaches[l];
            MemorySegment activeVCache = vCaches[l];
            if (!hasKV) {
                // Find the nearest previous layer that had K/V
                for (int prev = l - 1; prev >= 0; prev--) {
                    if (hasWeight("transformer.layer_" + prev + ".attn.k.w")) {
                        activeKCache = kCaches[prev];
                        activeVCache = vCaches[prev];
                        break;
                    }
                }
            }

            // ── Multi-Head Attention (GQA) ──
            // Q: [numQHeads, headDim], K/V cache: [seqLen, numKvHeads, headDim]
            int seqLen = pos + 1;
            for (int h = 0; h < NUM_Q_HEADS; h++) {
                long qOff = (long) h * HEAD_DIM;
                int kvHead = h / (NUM_Q_HEADS / NUM_KV_HEADS); // GQA: map Q head to KV head

                // Compute attention scores: score[t] = Q_h · K_t / sqrt(headDim)
                MemorySegment scores = a.allocate((long) seqLen * 4, 64);
                float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));

                for (int t = 0; t < seqLen; t++) {
                    float dot = 0;
                    long kOff = (long) t * KV_DIM + (long) kvHead * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dot += q.getAtIndex(ValueLayout.JAVA_FLOAT, qOff + d)
                                * activeKCache.getAtIndex(ValueLayout.JAVA_FLOAT, kOff + d);
                    }
                    scores.setAtIndex(ValueLayout.JAVA_FLOAT, t, dot * scale);
                }

                // Softmax
                softmax(scores, seqLen);

                // Weighted sum of V
                for (int d = 0; d < HEAD_DIM; d++) {
                    float sum = 0;
                    for (int t = 0; t < seqLen; t++) {
                        long vOff = (long) t * KV_DIM + (long) kvHead * HEAD_DIM + d;
                        sum += scores.getAtIndex(ValueLayout.JAVA_FLOAT, t)
                                * activeVCache.getAtIndex(ValueLayout.JAVA_FLOAT, vOff);
                    }
                    attnOut.setAtIndex(ValueLayout.JAVA_FLOAT, (long) h * HEAD_DIM + d, sum);
                }
            }

            // ── Output Projection (attn_vec_einsum) ──
            linearForward(projOut, attnOut, prefix + "attn.attn_vec_einsum", HIDDEN_DIM, Q_DIM, a);

            // ── Post-attention norm + residual ──
            if (hasWeight(prefix + "post_attention_norm.scale")) {
                rmsNorm(projOut, projOut, getWeight(prefix + "post_attention_norm.scale"), HIDDEN_DIM);
            }
            // Residual connection
            metal.add(x, residual, projOut, HIDDEN_DIM);

            // ── Pre-FFN RMSNorm ──
            rmsNorm(normed, x, getWeight(prefix + "pre_ffw_norm.scale"), HIDDEN_DIM);
            MemorySegment.copy(x, 0, residual, 0, (long) HIDDEN_DIM * 4);

            // ── SwiGLU FFN ──
            // gate = silu(normed @ ff_gate.w)
            // up   = normed @ ff1.w
            // ffn  = gate * up
            // down = ffn @ linear.w
            linearForward(gate, normed, prefix + "mlp.ff_gate", FFN_DIM, HIDDEN_DIM, a);
            linearForward(up, normed, prefix + "mlp.ff1", FFN_DIM, HIDDEN_DIM, a);
            metal.siluFfn(ffnOut, gate, up, FFN_DIM);
            linearForward(down, ffnOut, prefix + "mlp.linear", HIDDEN_DIM, FFN_DIM, a);

            // ── Post-FFN norm + residual ──
            if (hasWeight(prefix + "post_ffw_norm.scale")) {
                rmsNorm(down, down, getWeight(prefix + "post_ffw_norm.scale"), HIDDEN_DIM);
            }

            // Skip connection with learnable scale
            if (hasWeight(prefix + "skip.scale")) {
                float skipScale = getWeight(prefix + "skip.scale")
                        .getAtIndex(ValueLayout.JAVA_FLOAT, 0);
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    float r = residual.getAtIndex(ValueLayout.JAVA_FLOAT, i) * skipScale;
                    float d = down.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    x.setAtIndex(ValueLayout.JAVA_FLOAT, i, r + d);
                }
            } else {
                metal.add(x, residual, down, HIDDEN_DIM);
            }
        }

        // ── 3. FINAL NORM ──
        rmsNorm(x, x, getWeight("transformer.final_norm.scale"), HIDDEN_DIM);

        // ── 4. LM HEAD (weight-tied with embedding) ──
        // logits = x[1, hiddenDim] @ embeddingWeight^T[hiddenDim, vocabSize]
        // Since embedding is tied, we cache its dequantized float32 version.
        String embKey = "transformer.embedder.input_embedding.w_float32";
        MemorySegment embWFloat = dequantizedWeights.get(embKey);
        if (embWFloat == null) {
            int vSize = (int) (embW.byteSize() / HIDDEN_DIM);
            embWFloat = dequantizeInt8(embW, embScale, vSize, HIDDEN_DIM, arena);
            dequantizedWeights.put(embKey, embWFloat);
        }

        int vocabSize = (int) (embWFloat.byteSize() / (HIDDEN_DIM * 4));
        MemorySegment logits = a.allocate((long) vocabSize * 4, 64);
        
        // out[1, vocabSize] = x[1, hiddenDim] @ embWFloat^T[hiddenDim, vocabSize]
        metal.matmul(logits, x, embWFloat, 1, HIDDEN_DIM, vocabSize, 1.0f, 0.0f);

        return logits;
    }

    // ── Generation ──────────────────────────────────────────────────────────

    public void generate(String prompt, Consumer<String> tokenCallback) {
        if (!initialized) {
            throw new RuntimeException("Engine not initialized — call initialize() first");
        }

        log.info("Starting generation for prompt ({} chars)", prompt.length());

        int[] inputIds = tokenizer.encodeChatPrompt(prompt);
        int promptLen = inputIds.length;

        // Reset KV cache
        cacheLen = 0;

        int maxNewTokens = 512;
        int vocabSize = 0;

        try {
            // ── PREFILL: process all prompt tokens ──
            for (int i = 0; i < promptLen; i++) {
                try (Arena stepArena = Arena.ofConfined()) {
                    MemorySegment logits = forwardStep(inputIds[i], cacheLen, stepArena);
                    vocabSize = (int) (logits.byteSize() / 4);
                    cacheLen++;
                }
            }

            // ── DECODE: generate new tokens ──
            // Get the last logits for first prediction
            int nextToken;
            try (Arena stepArena = Arena.ofConfined()) {
                // Re-run last token to get logits (they were discarded)
                cacheLen--; // rewind
                MemorySegment logits = forwardStep(inputIds[promptLen - 1], cacheLen, stepArena);
                vocabSize = (int) (logits.byteSize() / 4);
                nextToken = argmax(logits, vocabSize);
                cacheLen++;
            }

            for (int i = 0; i < maxNewTokens; i++) {
                if (tokenizer.isEosToken(nextToken)) break;

                String tokenStr = tokenizer.decodeToken(nextToken);
                tokenCallback.accept(tokenStr);

                try (Arena stepArena = Arena.ofConfined()) {
                    MemorySegment logits = forwardStep(nextToken, cacheLen, stepArena);
                    nextToken = argmax(logits, vocabSize);
                    cacheLen++;
                }

                if (cacheLen >= MAX_SEQ_LEN - 1) {
                    log.warn("Reached max sequence length ({})", MAX_SEQ_LEN);
                    break;
                }
            }

        } catch (Exception e) {
            log.error("Generation failed at pos={}", cacheLen, e);
            tokenCallback.accept("\n[Error: " + e.getMessage() + "]");
        }
    }

    @Override
    public void close() {
        arena.close();
    }
}
