package tech.kayys.gollek.provider.litert;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
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
 * <p>
 * Uses {@link MetalBinding} for hardware-accelerated operations on Apple
 * Silicon
 * and automatically falls back to the CPU implementation on all other
 * platforms.
 * Weights are loaded from the companion {@code .task} file (MediaPipe format)
 * since {@code .litertlm} files use opaque NPU-compiled blobs.
 *
 * <p>
 * Architecture (Gemma-4 E2B):
 * <ul>
 * <li>35 transformer layers with GQA (8 Q heads : 1 KV head)</li>
 * <li>hidden_dim=1536, head_dim=256, ffn_dim=6144</li>
 * <li>INT4 quantized weights with per-channel float32 scales</li>
 * <li>Pre/post attention + FFN RMSNorm with skip connections</li>
 * <li>SwiGLU FFN: gate * silu(ff1) → down projection</li>
 * </ul>
 */
public class LiteRTGemmaMetalRunner implements AutoCloseable {

    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(LiteRTGemmaMetalRunner.class);

    private final MetalBinding metal;
    private final Path modelPath;
    private final LiteRTTokenizer tokenizer;

    private final Arena arena = Arena.ofShared();
    private Map<String, MemorySegment> weightSegments;
    private final Map<String, MemorySegment> dequantizedWeights = new java.util.concurrent.ConcurrentHashMap<>();
    private final Map<Integer, MemorySegment> lmHeadChunkCache = new ConcurrentHashMap<>();
    private long matmulCalls;
    private long rmsNormCalls;
    private long addCalls;
    private long dequantizeNanos;
    private long lmHeadNanos;
    private boolean initialized = false;

    // ── Architecture Constants (Gemma-4 E2B) ──────────────────────────────────
    private static final int NUM_LAYERS = 35;
    private static final int HIDDEN_DIM = 1536;
    private static final int PLE_DIM = 256;
    private static final int NUM_Q_HEADS = 8;
    private static final int NUM_KV_HEADS = 1;
    private static final int LOCAL_HEAD_DIM = 256;
    private static final int GLOBAL_HEAD_DIM = 512;
    private static final int MAX_HEAD_DIM = GLOBAL_HEAD_DIM;
    private static final int MAX_Q_DIM = NUM_Q_HEADS * MAX_HEAD_DIM;
    private static final int MAX_KV_DIM = NUM_KV_HEADS * MAX_HEAD_DIM;
    private static final int FFN_DIM = 6144;
    private static final int MAX_SEQ_LEN = 2048;
    private static final int SLIDING_WINDOW = 512;
    private static final int LM_HEAD_CHUNK_SIZE = 262144;
    private static final float EMBEDDING_SCALE = (float) Math.sqrt(HIDDEN_DIM);
    private static final float PLE_EMBEDDING_SCALE = (float) Math.sqrt(PLE_DIM);
    private static final float PER_LAYER_BLEND_SCALE = (float) (1.0 / Math.sqrt(2.0));
    private static final float PER_LAYER_MODEL_SCALE = (float) (1.0 / Math.sqrt(HIDDEN_DIM));
    private static final float FINAL_LOGIT_SOFTCAP = 30.0f;
    private static final float RMS_EPS = 1.0e-6f;
    private static final boolean DISABLE_GEMMA4_PLE = Boolean.getBoolean("gollek.litert.disable_gemma4_ple");
    private static final boolean DISABLE_GEMMA4_LAYER_SCALAR = Boolean
            .getBoolean("gollek.litert.disable_gemma4_layer_scalar");
    private static final boolean USE_LEGACY_SIGNED_ROW_MAJOR_INT4 = Boolean
            .getBoolean("gollek.litert.legacy_signed_row_major_int4");

    // TFLite type constants
    private static final int TFLITE_INT8 = 9;
    private static final int TFLITE_INT4 = 17;

    // ── KV Cache ──────────────────────────────────────────────────────────────
    // Per-layer KV cache: [seqLen, numKvHeads, headDim]
    private MemorySegment[] kCaches;
    private MemorySegment[] vCaches;
    private MemorySegment localUnitNormWeight;
    private MemorySegment globalUnitNormWeight;
    private int[] layerHeadDims;
    private int[] layerQDims;
    private int[] layerKVDims;
    private boolean[] fullAttentionLayers;
    private boolean[] ownKvLayers;
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
        if (fileName.endsWith(".task"))
            return modelPath;

        Path dir = modelPath.getParent();
        if (dir == null)
            dir = Path.of(".");

        String baseName = fileName.replaceAll("(_qualcomm_[^.]+)?\\.litertlm$", "");
        for (String suffix : new String[] { "-web.task", ".task" }) {
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
        if (!metal.isNativeAvailable()) {
            log.warn("Metal native bridge is unavailable; using checked CPU fallback kernels.");
        }

        // 2. Load weights from .task file
        Path weightSource = resolveWeightSource();
        Map<String, LiteRTContainerParser.WeightEntry> weightEntries = LiteRTContainerParser
                .extractWeightMap(weightSource);

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

        // 3. Discover per-layer layout and allocate KV caches
        initializeLayerLayouts();
        localUnitNormWeight = arena.allocate((long) LOCAL_HEAD_DIM * Float.BYTES, 64);
        globalUnitNormWeight = arena.allocate((long) GLOBAL_HEAD_DIM * Float.BYTES, 64);
        for (int i = 0; i < LOCAL_HEAD_DIM; i++) {
            localUnitNormWeight.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
        }
        for (int i = 0; i < GLOBAL_HEAD_DIM; i++) {
            globalUnitNormWeight.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
        }
        kCaches = new MemorySegment[NUM_LAYERS];
        vCaches = new MemorySegment[NUM_LAYERS];
        long totalKvBytes = 0L;
        for (int l = 0; l < NUM_LAYERS; l++) {
            long kvLayerSize = (long) MAX_SEQ_LEN * layerKVDims[l] * 4;
            kCaches[l] = arena.allocate(kvLayerSize, 64);
            vCaches[l] = arena.allocate(kvLayerSize, 64);
            totalKvBytes += kvLayerSize * 2;
        }
        cacheLen = 0;
        log.info("✓ KV-Cache allocated ({} MB total across {} layers)",
                totalKvBytes / (1024 * 1024), NUM_LAYERS);

        this.initialized = true;
    }

    private void initializeLayerLayouts() {
        layerHeadDims = new int[NUM_LAYERS];
        layerQDims = new int[NUM_LAYERS];
        layerKVDims = new int[NUM_LAYERS];
        fullAttentionLayers = new boolean[NUM_LAYERS];
        ownKvLayers = new boolean[NUM_LAYERS];

        for (int l = 0; l < NUM_LAYERS; l++) {
            String prefix = "transformer.layer_" + l + ".";
            int headDim = LOCAL_HEAD_DIM;
            if (hasWeight(prefix + "attn.q_norm.scale")) {
                headDim = (int) (getWeight(prefix + "attn.q_norm.scale").byteSize() / Float.BYTES);
            }
            layerHeadDims[l] = headDim;
            layerQDims[l] = NUM_Q_HEADS * headDim;
            layerKVDims[l] = NUM_KV_HEADS * headDim;
            fullAttentionLayers[l] = headDim > LOCAL_HEAD_DIM;
            ownKvLayers[l] = hasWeight(prefix + "attn.k.w");
        }
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
     *
     * <p>
     * The Gemma LiteRT companion `.task` weights are packed input-major and use
     * centered unsigned nibbles (q - 8) with one scale per output channel. We
     * expand them here into a conventional row-major [outDim, inDim] matrix so
     * the rest of the runner can keep using standard GEMM.
     *
     * <p>
     * The legacy row-major signed-nibble path remains behind a property only
     * as a diagnostics escape hatch.
     */
    private MemorySegment dequantizeInt4(MemorySegment weight, MemorySegment scale,
            int outDim, int inDim, Arena a) {
        long floatSize = (long) outDim * inDim * 4;
        MemorySegment out = a.allocate(floatSize, 64);

        byte[] wArr = weight.toArray(ValueLayout.JAVA_BYTE);
        float[] sArr = scale.toArray(ValueLayout.JAVA_FLOAT);

        if (USE_LEGACY_SIGNED_ROW_MAJOR_INT4) {
            for (int row = 0; row < outDim; row++) {
                float s = sArr[row];
                int rowByteOffset = row * inDim / 2;
                long outRowOffset = (long) row * inDim;

                for (int col = 0; col < inDim; col += 2) {
                    byte packed = wArr[rowByteOffset + col / 2];
                    int lo = (packed & 0x0F);
                    int hi = (packed >> 4) & 0x0F;
                    if (lo >= 8)
                        lo -= 16;
                    if (hi >= 8)
                        hi -= 16;

                    out.setAtIndex(ValueLayout.JAVA_FLOAT, outRowOffset + col, lo * s);
                    if (col + 1 < inDim) {
                        out.setAtIndex(ValueLayout.JAVA_FLOAT, outRowOffset + col + 1, hi * s);
                    }
                }
            }
            return out;
        }

        for (int in = 0; in < inDim; in++) {
            long baseFlatIndex = (long) in * outDim;
            for (int outRow = 0; outRow < outDim; outRow++) {
                long flatIndex = baseFlatIndex + outRow;
                int packed = wArr[(int) (flatIndex >>> 1)] & 0xFF;
                int q = ((flatIndex & 1L) == 0L) ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
                q -= 8;
                out.setAtIndex(ValueLayout.JAVA_FLOAT,
                        (long) outRow * inDim + in,
                        q * sArr[outRow]);
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

    private void lookupScaledEmbeddingRow(
            MemorySegment out,
            MemorySegment packedWeight,
            MemorySegment scale,
            int tokenId,
            int dim,
            float embeddingScale) {
        int vocabSize = (int) (scale.byteSize() / Float.BYTES);
        if (tokenId < 0 || tokenId >= vocabSize) {
            throw new IllegalArgumentException("Token id " + tokenId + " out of range for vocab " + vocabSize);
        }

        long rowBytes = packedWeight.byteSize() / vocabSize;
        if (rowBytes * vocabSize != packedWeight.byteSize()) {
            throw new IllegalStateException("Embedding table size does not divide cleanly by vocab");
        }

        float rowScale = scale.getAtIndex(ValueLayout.JAVA_FLOAT, tokenId);
        if (rowBytes == dim) {
            long rowOffset = (long) tokenId * dim;
            for (int i = 0; i < dim; i++) {
                byte val = packedWeight.get(ValueLayout.JAVA_BYTE, rowOffset + i);
                out.setAtIndex(ValueLayout.JAVA_FLOAT, i, val * rowScale * embeddingScale);
            }
            return;
        }

        if (rowBytes * 2 == dim) {
            long rowOffset = (long) tokenId * rowBytes;
            for (int i = 0; i < dim; i += 2) {
                int packed = packedWeight.get(ValueLayout.JAVA_BYTE, rowOffset + (i / 2)) & 0xFF;
                int lo = packed & 0x0F;
                int hi = (packed >>> 4) & 0x0F;
                if (lo >= 8)
                    lo -= 16;
                if (hi >= 8)
                    hi -= 16;
                out.setAtIndex(ValueLayout.JAVA_FLOAT, i, lo * rowScale * embeddingScale);
                if (i + 1 < dim) {
                    out.setAtIndex(ValueLayout.JAVA_FLOAT, i + 1, hi * rowScale * embeddingScale);
                }
            }
            return;
        }

        if (rowBytes * 4 == dim) {
            long rowOffset = (long) tokenId * rowBytes;
            for (int i = 0; i < dim; i++) {
                int packed = packedWeight.get(ValueLayout.JAVA_BYTE, rowOffset + (i / 4)) & 0xFF;
                int shift = (i & 0x03) * 2;
                float q = ((packed >>> shift) & 0x03) - 1.5f;
                out.setAtIndex(ValueLayout.JAVA_FLOAT, i, q * rowScale * embeddingScale);
            }
            return;
        }

        throw new IllegalStateException("Unsupported embedding row layout: rowBytes=" + rowBytes + " dim=" + dim);
    }

    private void applyLayerNormPerSlice(MemorySegment packed, MemorySegment normWeight, int slices, int sliceDim) {
        for (int i = 0; i < slices; i++) {
            long offset = (long) i * sliceDim * Float.BYTES;
            rmsNorm(packed.asSlice(offset, (long) sliceDim * Float.BYTES),
                    packed.asSlice(offset, (long) sliceDim * Float.BYTES),
                    normWeight,
                    sliceDim);
        }
    }

    private void combinePerLayerInput(
            MemorySegment out,
            int layerIndex,
            int tokenId,
            MemorySegment projectedPerLayerPacked) {
        String prefix = "transformer.layer_" + layerIndex + ".";
        lookupScaledEmbeddingRow(
                out,
                getWeight(prefix + "per_layer_embeddings.w"),
                getWeight(prefix + "per_layer_embeddings.w_quantized_scale"),
                tokenId,
                PLE_DIM,
                PLE_EMBEDDING_SCALE);

        long projectedOffset = (long) layerIndex * PLE_DIM * Float.BYTES;
        MemorySegment projectedSlice = projectedPerLayerPacked.asSlice(projectedOffset, (long) PLE_DIM * Float.BYTES);
        for (int i = 0; i < PLE_DIM; i++) {
            float combined = (out.getAtIndex(ValueLayout.JAVA_FLOAT, i)
                    + projectedSlice.getAtIndex(ValueLayout.JAVA_FLOAT, i)) * PER_LAYER_BLEND_SCALE;
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, combined);
        }
    }

    private void geluTanhInPlace(MemorySegment values, int dim) {
        final float sqrt2OverPi = 0.7978845608f;
        final float coeff = 0.044715f;
        for (int i = 0; i < dim; i++) {
            float x = values.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float x3 = x * x * x;
            float inner = sqrt2OverPi * (x + coeff * x3);
            float gelu = 0.5f * x * (1.0f + (float) Math.tanh(inner));
            values.setAtIndex(ValueLayout.JAVA_FLOAT, i, gelu);
        }
    }

    private void geluMultiply(MemorySegment out, MemorySegment gate, MemorySegment up, int dim) {
        final float sqrt2OverPi = 0.7978845608f;
        final float coeff = 0.044715f;
        for (int i = 0; i < dim; i++) {
            float x = gate.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float x3 = x * x * x;
            float inner = sqrt2OverPi * (x + coeff * x3);
            float gelu = 0.5f * x * (1.0f + (float) Math.tanh(inner));
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, gelu * up.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        }
    }

    private void applyRotaryEmbedding(MemorySegment qk, int headDim, int pos, boolean fullAttention) {
        int rotaryDim = fullAttention ? headDim / 4 : headDim;
        float theta = fullAttention ? 1_000_000.0f : 10_000.0f;
        int half = rotaryDim / 2;
        for (int i = 0; i < half; i++) {
            float freq = (float) (1.0 / Math.pow(theta, (double) i / half));
            float angle = freq * pos;
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);

            float x1 = qk.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float x2 = qk.getAtIndex(ValueLayout.JAVA_FLOAT, i + half);
            qk.setAtIndex(ValueLayout.JAVA_FLOAT, i, x1 * cos - x2 * sin);
            qk.setAtIndex(ValueLayout.JAVA_FLOAT, i + half, x2 * cos + x1 * sin);
        }
    }

    private void applyFinalLogitSoftcap(MemorySegment logits, int size) {
        for (int i = 0; i < size; i++) {
            float value = logits.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float softcapped = (float) Math.tanh(value / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP;
            logits.setAtIndex(ValueLayout.JAVA_FLOAT, i, softcapped);
        }
    }

    private MemorySegment dequantizeEmbeddingChunk(int startRow, int rowCount) {
        return lmHeadChunkCache.computeIfAbsent(startRow, ignored -> {
            long startNanos = System.nanoTime();
            MemorySegment packedWeight = getWeight("transformer.embedder.input_embedding.w");
            MemorySegment scale = getWeight("transformer.embedder.input_embedding.w_quantized_scale");
            MemorySegment chunk = arena.allocate((long) rowCount * HIDDEN_DIM * Float.BYTES, 64);
            MemorySegment rowBuffer = arena.allocate((long) HIDDEN_DIM * Float.BYTES, 64);
            for (int row = 0; row < rowCount; row++) {
                lookupScaledEmbeddingRow(rowBuffer, packedWeight, scale, startRow + row, HIDDEN_DIM, 1.0f);
                MemorySegment.copy(rowBuffer, 0, chunk, (long) row * HIDDEN_DIM * Float.BYTES,
                        (long) HIDDEN_DIM * Float.BYTES);
            }
            dequantizeNanos += System.nanoTime() - startNanos;
            return chunk;
        });
    }

    private int argmaxLogitsFromEmbeddingHead(MemorySegment hiddenState, Arena stepArena) {
        long startNanos = System.nanoTime();
        MemorySegment scale = getWeight("transformer.embedder.input_embedding.w_quantized_scale");
        int vocabSize = (int) (scale.byteSize() / Float.BYTES);
        int bestToken = 0;
        float bestLogit = Float.NEGATIVE_INFINITY;

        for (int start = 0; start < vocabSize; start += LM_HEAD_CHUNK_SIZE) {
            int rows = Math.min(LM_HEAD_CHUNK_SIZE, vocabSize - start);
            MemorySegment chunkWeights = dequantizeEmbeddingChunk(start, rows);
            MemorySegment logits = stepArena.allocate((long) rows * Float.BYTES, 64);
            matmulTransposedRightChecked(logits, hiddenState, chunkWeights, HIDDEN_DIM, rows, "lm_head_chunk_" + start);
            applyFinalLogitSoftcap(logits, rows);
            for (int i = 0; i < rows; i++) {
                float value = logits.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                if (value > bestLogit) {
                    bestLogit = value;
                    bestToken = start + i;
                }
            }
        }
        lmHeadNanos += System.nanoTime() - startNanos;
        return bestToken;
    }

    // ── Core Operations ─────────────────────────────────────────────────────

    /** RMS Normalization: out = x * weight / rms(x) */
    private void rmsNorm(MemorySegment out, MemorySegment x, MemorySegment weight, int dim) {
        int status = metal.rmsNorm(out, x, weight, dim, RMS_EPS, false);
        if (status != 0) {
            throw new IllegalStateException("Metal/CPU rmsNorm failed with status " + status + " for dim=" + dim);
        }
        rmsNormCalls++;
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

        // Cached weights are row-major [outDim, inDim], so the right-hand side is
        // transposed.
        matmulTransposedRightChecked(out, input, wFloat, inDim, outDim, weightName);
    }

    private void matmulTransposedRightChecked(
            MemorySegment out,
            MemorySegment input,
            MemorySegment rowMajorWeight,
            int inDim,
            int outDim,
            String label) {
        int status = metal.matmulTransposedRight(out, input, rowMajorWeight, 1, inDim, outDim, 1.0f, 0.0f);
        if (status != 0) {
            throw new IllegalStateException("Metal/CPU matmulTransposedRight failed with status "
                    + status + " for " + label + " [1," + inDim + "] x [" + outDim + "," + inDim + "]^T");
        }
        matmulCalls++;
    }

    private void addChecked(MemorySegment out, MemorySegment left, MemorySegment right, int dim, String label) {
        int status = metal.add(out, left, right, dim);
        if (status != 0) {
            throw new IllegalStateException("Metal/CPU add failed with status " + status + " for " + label);
        }
        addCalls++;
    }

    /** Softmax in-place over logits[0..size-1]. */
    private void softmax(MemorySegment logits, int size) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            float v = logits.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (v > max)
                max = v;
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

    // ── Transformer Forward Pass ────────────────────────────────────────────

    private MemorySegment forwardHiddenState(int tokenId, int pos, Arena a) {
        MemorySegment x = a.allocate((long) HIDDEN_DIM * Float.BYTES, 64);
        lookupScaledEmbeddingRow(
                x,
                getWeight("transformer.embedder.input_embedding.w"),
                getWeight("transformer.embedder.input_embedding.w_quantized_scale"),
                tokenId,
                HIDDEN_DIM,
                EMBEDDING_SCALE);

        MemorySegment projectedPerLayer = a.allocate((long) NUM_LAYERS * PLE_DIM * Float.BYTES, 64);
        linearForward(
                projectedPerLayer,
                x,
                "transformer.embedder.per_layer_model_projection",
                NUM_LAYERS * PLE_DIM,
                HIDDEN_DIM,
                a);
        for (int i = 0; i < NUM_LAYERS * PLE_DIM; i++) {
            projectedPerLayer.setAtIndex(
                    ValueLayout.JAVA_FLOAT,
                    i,
                    projectedPerLayer.getAtIndex(ValueLayout.JAVA_FLOAT, i) * PER_LAYER_MODEL_SCALE);
        }
        applyLayerNormPerSlice(
                projectedPerLayer,
                getWeight("transformer.embedder.per_layer_projection_norm.scale"),
                NUM_LAYERS,
                PLE_DIM);

        MemorySegment residual = a.allocate((long) HIDDEN_DIM * Float.BYTES, 64);
        MemorySegment normed = a.allocate((long) HIDDEN_DIM * Float.BYTES, 64);
        MemorySegment q = a.allocate((long) MAX_Q_DIM * Float.BYTES, 64);
        MemorySegment k = a.allocate((long) MAX_KV_DIM * Float.BYTES, 64);
        MemorySegment v = a.allocate((long) MAX_KV_DIM * Float.BYTES, 64);
        MemorySegment attnOut = a.allocate((long) MAX_Q_DIM * Float.BYTES, 64);
        MemorySegment projOut = a.allocate((long) HIDDEN_DIM * Float.BYTES, 64);
        MemorySegment gate = a.allocate((long) FFN_DIM * Float.BYTES, 64);
        MemorySegment up = a.allocate((long) FFN_DIM * Float.BYTES, 64);
        MemorySegment ffnOut = a.allocate((long) FFN_DIM * Float.BYTES, 64);
        MemorySegment down = a.allocate((long) HIDDEN_DIM * Float.BYTES, 64);
        MemorySegment perLayerInput = a.allocate((long) PLE_DIM * Float.BYTES, 64);
        MemorySegment perLayerGate = a.allocate((long) PLE_DIM * Float.BYTES, 64);
        MemorySegment perLayerProj = a.allocate((long) HIDDEN_DIM * Float.BYTES, 64);

        int lastSlidingKvLayer = -1;
        int lastFullKvLayer = -1;

        for (int l = 0; l < NUM_LAYERS; l++) {
            String prefix = "transformer.layer_" + l + ".";
            int headDim = layerHeadDims[l];
            int qDim = layerQDims[l];
            int kvDim = layerKVDims[l];
            boolean fullAttention = fullAttentionLayers[l];

            rmsNorm(normed, x, getWeight(prefix + "pre_attention_norm.scale"), HIDDEN_DIM);
            MemorySegment.copy(x, 0, residual, 0, (long) HIDDEN_DIM * Float.BYTES);

            linearForward(q, normed, prefix + "attn.q", qDim, HIDDEN_DIM, a);
            if (hasWeight(prefix + "attn.q_norm.scale")) {
                MemorySegment qNormW = getWeight(prefix + "attn.q_norm.scale");
                for (int h = 0; h < NUM_Q_HEADS; h++) {
                    long headOffset = (long) h * headDim * Float.BYTES;
                    rmsNorm(q.asSlice(headOffset, (long) headDim * Float.BYTES),
                            q.asSlice(headOffset, (long) headDim * Float.BYTES),
                            qNormW,
                            headDim);
                }
            }
            for (int h = 0; h < NUM_Q_HEADS; h++) {
                long headOffset = (long) h * headDim * Float.BYTES;
                applyRotaryEmbedding(q.asSlice(headOffset, (long) headDim * Float.BYTES), headDim, pos, fullAttention);
            }

            if (ownKvLayers[l]) {
                linearForward(k, normed, prefix + "attn.k", kvDim, HIDDEN_DIM, a);
                linearForward(v, normed, prefix + "attn.v", kvDim, HIDDEN_DIM, a);
                if (hasWeight(prefix + "attn.k_norm.scale")) {
                    MemorySegment kNormW = getWeight(prefix + "attn.k_norm.scale");
                    rmsNorm(k, k, kNormW, kvDim);
                }
                rmsNorm(v, v, fullAttention ? globalUnitNormWeight : localUnitNormWeight, kvDim);
                applyRotaryEmbedding(k, headDim, pos, fullAttention);
                MemorySegment.copy(k, 0, kCaches[l], (long) pos * kvDim * Float.BYTES, (long) kvDim * Float.BYTES);
                MemorySegment.copy(v, 0, vCaches[l], (long) pos * kvDim * Float.BYTES, (long) kvDim * Float.BYTES);
                if (fullAttention) {
                    lastFullKvLayer = l;
                } else {
                    lastSlidingKvLayer = l;
                }
            }

            int sourceLayer = ownKvLayers[l]
                    ? l
                    : (fullAttention ? lastFullKvLayer : lastSlidingKvLayer);
            if (sourceLayer < 0) {
                throw new IllegalStateException("No KV source layer available for layer " + l);
            }

            MemorySegment activeKCache = kCaches[sourceLayer];
            MemorySegment activeVCache = vCaches[sourceLayer];
            int startPos = fullAttention ? 0 : Math.max(0, pos + 1 - SLIDING_WINDOW);
            int scoreLen = pos - startPos + 1;
            // Gemma 4 text attention uses an unscaled dot product in the HF reference.
            float attentionScale = 1.0f;

            for (int h = 0; h < NUM_Q_HEADS; h++) {
                long qOffset = (long) h * headDim;
                int kvHead = h / (NUM_Q_HEADS / NUM_KV_HEADS);
                MemorySegment scores = a.allocate((long) scoreLen * Float.BYTES, 64);

                for (int idx = 0, t = startPos; idx < scoreLen; idx++, t++) {
                    float dot = 0.0f;
                    long kOffset = (long) t * kvDim + (long) kvHead * headDim;
                    for (int d = 0; d < headDim; d++) {
                        dot += q.getAtIndex(ValueLayout.JAVA_FLOAT, qOffset + d)
                                * activeKCache.getAtIndex(ValueLayout.JAVA_FLOAT, kOffset + d);
                    }
                    scores.setAtIndex(ValueLayout.JAVA_FLOAT, idx, dot * attentionScale);
                }

                softmax(scores, scoreLen);
                for (int d = 0; d < headDim; d++) {
                    float sum = 0.0f;
                    for (int idx = 0, t = startPos; idx < scoreLen; idx++, t++) {
                        long vOffset = (long) t * kvDim + (long) kvHead * headDim + d;
                        sum += scores.getAtIndex(ValueLayout.JAVA_FLOAT, idx)
                                * activeVCache.getAtIndex(ValueLayout.JAVA_FLOAT, vOffset);
                    }
                    attnOut.setAtIndex(ValueLayout.JAVA_FLOAT, (long) h * headDim + d, sum);
                }
            }

            linearForward(projOut, attnOut, prefix + "attn.attn_vec_einsum", HIDDEN_DIM, qDim, a);
            if (hasWeight(prefix + "post_attention_norm.scale")) {
                rmsNorm(projOut, projOut, getWeight(prefix + "post_attention_norm.scale"), HIDDEN_DIM);
            }
            addChecked(x, residual, projOut, HIDDEN_DIM, prefix + "attention_residual");

            rmsNorm(normed, x, getWeight(prefix + "pre_ffw_norm.scale"), HIDDEN_DIM);
            MemorySegment.copy(x, 0, residual, 0, (long) HIDDEN_DIM * Float.BYTES);
            linearForward(gate, normed, prefix + "mlp.ff_gate", FFN_DIM, HIDDEN_DIM, a);
            linearForward(up, normed, prefix + "mlp.ff1", FFN_DIM, HIDDEN_DIM, a);
            geluMultiply(ffnOut, gate, up, FFN_DIM);
            linearForward(down, ffnOut, prefix + "mlp.linear", HIDDEN_DIM, FFN_DIM, a);
            if (hasWeight(prefix + "post_ffw_norm.scale")) {
                rmsNorm(down, down, getWeight(prefix + "post_ffw_norm.scale"), HIDDEN_DIM);
            }

            addChecked(x, residual, down, HIDDEN_DIM, prefix + "ffn_residual");

            if (!DISABLE_GEMMA4_PLE && hasWeight(prefix + "per_layer_embedding_gate.w")) {
                combinePerLayerInput(perLayerInput, l, tokenId, projectedPerLayer);
                MemorySegment.copy(x, 0, residual, 0, (long) HIDDEN_DIM * Float.BYTES);
                linearForward(perLayerGate, x, prefix + "per_layer_embedding_gate", PLE_DIM, HIDDEN_DIM, a);
                geluTanhInPlace(perLayerGate, PLE_DIM);
                for (int i = 0; i < PLE_DIM; i++) {
                    perLayerGate.setAtIndex(
                            ValueLayout.JAVA_FLOAT,
                            i,
                            perLayerGate.getAtIndex(ValueLayout.JAVA_FLOAT, i)
                                    * perLayerInput.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                }
                linearForward(perLayerProj, perLayerGate, prefix + "per_layer_embedding_projection", HIDDEN_DIM,
                        PLE_DIM, a);
                if (hasWeight(prefix + "post_per_layer_input_norm.scale")) {
                    rmsNorm(perLayerProj, perLayerProj, getWeight(prefix + "post_per_layer_input_norm.scale"),
                            HIDDEN_DIM);
                }
                addChecked(x, residual, perLayerProj, HIDDEN_DIM, prefix + "per_layer_residual");
            }

            if (!DISABLE_GEMMA4_LAYER_SCALAR && hasWeight(prefix + "skip.scale")) {
                float layerScalar = getWeight(prefix + "skip.scale").getAtIndex(ValueLayout.JAVA_FLOAT, 0);
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    x.setAtIndex(ValueLayout.JAVA_FLOAT, i, x.getAtIndex(ValueLayout.JAVA_FLOAT, i) * layerScalar);
                }
            }
        }

        rmsNorm(x, x, getWeight("transformer.final_norm.scale"), HIDDEN_DIM);
        return x;
    }

    // ── Generation ──────────────────────────────────────────────────────────

    public void generate(String prompt, Consumer<String> tokenCallback) {
        generate(prompt, 512, tokenCallback);
    }

    public void generate(String prompt, int maxNewTokens, Consumer<String> tokenCallback) {
        if (!initialized) {
            throw new RuntimeException("Engine not initialized — call initialize() first");
        }

        String safePrompt = prompt == null ? "" : prompt;
        log.info("Starting generation for prompt ({} chars)", safePrompt.length());

        int[] inputIds = tokenizer.encodeChatPrompt(safePrompt);
        int promptLen = inputIds.length;

        // Reset KV cache
        cacheLen = 0;

        int nextToken = -1;

        try {
            for (int i = 0; i < promptLen - 1; i++) {
                try (Arena stepArena = Arena.ofConfined()) {
                    forwardHiddenState(inputIds[i], cacheLen, stepArena);
                    cacheLen++;
                }
            }

            try (Arena stepArena = Arena.ofConfined()) {
                MemorySegment hiddenState = forwardHiddenState(inputIds[promptLen - 1], cacheLen, stepArena);
                nextToken = argmaxLogitsFromEmbeddingHead(hiddenState, stepArena);
                cacheLen++;
            }

            for (int i = 0; i < maxNewTokens; i++) {
                if (tokenizer.isTerminalToken(nextToken))
                    break;

                String tokenStr = tokenizer.decodeToken(nextToken);
                if (!tokenStr.isEmpty()) {
                    tokenCallback.accept(tokenStr);
                }

                try (Arena stepArena = Arena.ofConfined()) {
                    MemorySegment hiddenState = forwardHiddenState(nextToken, cacheLen, stepArena);
                    nextToken = argmaxLogitsFromEmbeddingHead(hiddenState, stepArena);
                    cacheLen++;
                }

                if (cacheLen >= MAX_SEQ_LEN - 1) {
                    log.warn("Reached max sequence length ({})", MAX_SEQ_LEN);
                    break;
                }
            }
            log.info(
                    "Gemma fallback runner stats: matmulCalls={}, rmsNormCalls={}, addCalls={}, lmHeadMs={}, dequantizeMs={}, lmHeadChunks={}",
                    matmulCalls,
                    rmsNormCalls,
                    addCalls,
                    lmHeadNanos / 1_000_000,
                    dequantizeNanos / 1_000_000,
                    lmHeadChunkCache.size());

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
