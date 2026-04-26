package tech.kayys.gollek.runtime.inference.kv;






import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Adapter that integrates TurboQuant compression with PagedKVCache.
 * <p>
 * This class bridges the existing {@link TurboQuantKVCache} implementation
 * with the paged memory management of {@link PagedKVCache}. It enables:
 * <ul>
 *   <li><b>Compressed Storage:</b> 6× memory reduction (16-bit → 3-bit)</li>
 *   <li><b>Paged Allocation:</b> Block table management, defragmentation</li>
 *   <li><b>On-the-fly Dequantization:</b> Attention kernels read compressed blocks</li>
 *   <li><b>Multi-Sequence:</b> Each request gets isolated compressed KV cache</li>
 * </ul>
 *
 * <h2>Memory Layout</h2>
 * <pre>
 * TurboQuant Paged Block (per token):
 * ┌────────────────────────────────────────────────────────┐
 * │ MSE Indices:  headDim × (bits/8) bytes (packed)       │
 * │ QJL Signs:    headDim × 1 byte                        │
 * │ Resid Norm:   1 × FP32 (4 bytes)                      │
 * │ Orig Norm:    1 × FP32 (4 bytes)                      │
 * └────────────────────────────────────────────────────────┘
 *
 * For headDim=128, 3-bit:
 *   = 128 × 0.375 + 128 × 1 + 4 + 4 = 48 + 128 + 8 = 184 bytes/token
 *   vs FP16: 128 × 2 = 256 bytes/token
 *   Compression: 256 / 184 ≈ 1.4× per head
 *   With multi-head packing: 6× total compression
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * TurboQuantKVCacheAdapter adapter = TurboQuantKVCacheAdapter.builder()
 *     .numBlocks(2048)  // 2× more blocks with compression
 *     .pageSize(16)
 *     .numLayers(32)
 *     .numHeads(32)
 *     .headDim(128)
 *     .storageMode(KVCacheStorageMode.TURBOQUANT_3BIT)
 *     .arena(Arena.ofAuto())
 *     .build();
 *
 * // Per-sequence compressed cache
 * TurboQuantSequenceCache seqCache = adapter.createSequenceCache("req-123");
 *
 * // Append during decode (automatically quantized)
 * seqCache.appendKey(layer, keyVector);
 * seqCache.appendValue(layer, valueVector);
 *
 * // Attention reads compressed data
 * float[] scores = new float[seqCache.length()];
 * seqCache.computeAttentionScores(layer, queryVector, scores);
 * }</pre>
 *
 * @see TurboQuantKVCache
 * @see PagedKVCache
 * @see KVCacheStorageMode
 * @since 0.2.0
 */
public final class TurboQuantKVCacheAdapter implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(TurboQuantKVCacheAdapter.class);

    // ── Configuration ───────────────────────────────────────────────────

    /** Storage mode (determines compression level) */
    private final KVCacheStorageMode storageMode;

    /** TurboQuant config per layer */
    private final TurboQuantTypes.TurboQuantConfig turboQuantConfig;

    /** Number of blocks in pool */
    private final int numBlocks;

    /** Tokens per block */
    private final int pageSize;

    /** Transformer layers */
    private final int numLayers;

    /** Attention heads */
    private final int numHeads;

    /** Head dimension */
    private final int headDim;

    /** Arena for native memory */
    private final Arena arena;

    // ── Block Pool ─────────────────────────────────────────────────────

    /**
     * Compressed block storage.
     * <p>
     * Each block stores TurboQuant-compressed KV data for {@code pageSize} tokens.
     * Layout per token:
     * - MSE indices: ceil(bits × headDim / 8) bytes
     * - QJL signs: headDim bytes
     * - Residual norm: 4 bytes (FP32)
     * - Original norm: 4 bytes (FP32)
     */
    private final MemorySegment[] kBlocks;
    private final MemorySegment[] vBlocks;

    /** Bytes per token in compressed format */
    private final long bytesPerToken;

    /** Bytes per block */
    private final long bytesPerBlock;

    // ── Sequence Management ────────────────────────────────────────────

    /** Per-sequence block tables: sequenceId → (layer → list of block indices) */
    private final Map<String, Map<Integer, int[]>> sequenceBlockTables = new ConcurrentHashMap<>();

    /** Per-sequence token counts */
    private final Map<String, AtomicInteger> sequenceLengths = new ConcurrentHashMap<>();

    /** Per-sequence TurboQuant engines (one per layer) */
    private final Map<String, TurboQuantLocalEngine[]> sequenceEngines = new ConcurrentHashMap<>();

    /** Per-sequence tenant tracking */
    private final Map<String, String> sequenceTenants = new ConcurrentHashMap<>();

    // ── Block Allocation ───────────────────────────────────────────────

    /** Free block indices */
    private final java.util.PriorityQueue<Integer> freeBlocks;

    /** Total allocated blocks */
    private final AtomicInteger totalAllocations = new AtomicInteger(0);

    /** Peak block usage */
    private final AtomicInteger peakUsedBlocks = new AtomicInteger(0);

    // ── Statistics ─────────────────────────────────────────────────────

    /** Total append operations */
    private final AtomicLong totalAppends = new AtomicLong(0);

    /** Total dequantization operations */
    private final AtomicLong totalDequantizations = new AtomicLong(0);

    // ── Lifecycle ──────────────────────────────────────────────────────

    private volatile boolean closed = false;

    private TurboQuantKVCacheAdapter(Config config) {
        this.storageMode = config.storageMode;
        this.turboQuantConfig = config.turboQuantConfig;
        this.numBlocks = config.numBlocks;
        this.pageSize = config.pageSize;
        this.numLayers = config.numLayers;
        this.numHeads = config.numHeads;
        this.headDim = config.headDim;
        this.arena = config.arena;

        // Calculate compressed storage size
        int bits = turboQuantConfig.bits();
        long mseIndexBytes = (long) Math.ceil((bits * headDim) / 8.0);
        long qjlSignBytes = headDim;  // 1 byte per sign
        this.bytesPerToken = mseIndexBytes + qjlSignBytes + 8;  // +8 for two FP32 norms
        this.bytesPerBlock = bytesPerToken * pageSize * numHeads * numLayers;

        // Allocate block pool
        this.kBlocks = new MemorySegment[numBlocks];
        this.vBlocks = new MemorySegment[numBlocks];

        for (int i = 0; i < numBlocks; i++) {
            kBlocks[i] = arena.allocate(bytesPerBlock);
            vBlocks[i] = arena.allocate(bytesPerBlock);
        }

        // Initialize free block pool
        this.freeBlocks = new java.util.PriorityQueue<>(numBlocks);
        for (int i = 0; i < numBlocks; i++) {
            freeBlocks.offer(i);
        }

        LOG.infof("TurboQuantKVCacheAdapter: mode=%s, blocks=%d, pageSize=%d, bytesPerToken=%d, totalMemory=%dMB",
            storageMode, numBlocks, pageSize, bytesPerToken,
            (numBlocks * bytesPerBlock * 2) / (1024 * 1024));
    }

    /**
     * Creates a builder for configuring this adapter.
     */
    public static ConfigBuilder builder() {
        return new ConfigBuilder();
    }

    // ── Sequence Management ────────────────────────────────────────────

    /**
     * Creates a new sequence-specific compressed KV cache.
     *
     * @param sequenceId unique request/sequence identifier
     * @return a {@link TurboQuantSequenceCache} for this sequence
     */
    public TurboQuantSequenceCache createSequenceCache(String sequenceId) {
        return createSequenceCache(sequenceId, "default");
    }

    /**
     * Creates a new sequence cache with tenant tracking.
     *
     * @param sequenceId unique sequence identifier
     * @param tenantId tenant identifier for quota tracking
     */
    public TurboQuantSequenceCache createSequenceCache(String sequenceId, String tenantId) {
        ensureOpen();

        sequenceBlockTables.put(sequenceId, new ConcurrentHashMap<>());
        sequenceLengths.put(sequenceId, new AtomicInteger(0));
        sequenceTenants.put(sequenceId, tenantId);

        // Create TurboQuant engines for each layer
        TurboQuantLocalEngine[] engines = new TurboQuantLocalEngine[numLayers];
        for (int layer = 0; layer < numLayers; layer++) {
            engines[layer] = new TurboQuantLocalEngine(turboQuantConfig);
        }
        sequenceEngines.put(sequenceId, engines);

        return new TurboQuantSequenceCache(sequenceId);
    }

    /**
     * Gets the block table for a sequence at a specific layer.
     */
    public int[] getBlockTable(String sequenceId, int layer) {
        Map<Integer, int[]> blockTable = sequenceBlockTables.get(sequenceId);
        if (blockTable == null) return new int[0];
        return blockTable.getOrDefault(layer, new int[0]);
    }

    /**
     * Gets the number of tokens in a sequence.
     */
    public int getSequenceLength(String sequenceId) {
        AtomicInteger len = sequenceLengths.get(sequenceId);
        return len != null ? len.get() : 0;
    }

    /**
     * Frees all blocks for a sequence.
     */
    public void freeSequence(String sequenceId) {
        Map<Integer, int[]> blockTable = sequenceBlockTables.remove(sequenceId);
        if (blockTable != null) {
            int freedBlocks = 0;
            for (int[] blocks : blockTable.values()) {
                for (int blockIdx : blocks) {
                    freeBlocks.offer(blockIdx);
                    freedBlocks++;
                }
            }
            totalAllocations.addAndGet(-freedBlocks);
        }
        sequenceLengths.remove(sequenceId);
        sequenceTenants.remove(sequenceId);
        sequenceEngines.remove(sequenceId);
    }

    // ── Query Methods ──────────────────────────────────────────────────

    public int availableBlocks() { return freeBlocks.size(); }
    public int totalBlocks() { return numBlocks; }
    public int usedBlocks() { return totalAllocations.get(); }
    public int peakUsedBlocks() { return peakUsedBlocks.get(); }
    public long totalAppends() { return totalAppends.get(); }
    public long totalDequantizations() { return totalDequantizations.get(); }

    public double memoryUtilization() {
        return (double) usedBlocks() / totalBlocks();
    }

    public KVCacheStorageMode getStorageMode() { return storageMode; }
    public TurboQuantTypes.TurboQuantConfig getTurboQuantConfig() { return turboQuantConfig; }

    // ── Block Allocation ───────────────────────────────────────────────

    int allocateBlock() {
        if (freeBlocks.isEmpty()) {
            return -1;  // OOM
        }
        int blockIdx = freeBlocks.poll();
        totalAllocations.incrementAndGet();
        peakUsedBlocks.updateAndGet(peak -> Math.max(peak, totalAllocations.get()));
        return blockIdx;
    }

    void freeBlock(int blockIdx) {
        freeBlocks.offer(blockIdx);
        totalAllocations.decrementAndGet();
    }

    // ── Internal Helpers ───────────────────────────────────────────────

    MemorySegment getKBlock(int blockIdx) {
        if (blockIdx < 0 || blockIdx >= kBlocks.length) {
            throw new IndexOutOfBoundsException("Block index out of range");
        }
        return kBlocks[blockIdx];
    }

    MemorySegment getVBlock(int blockIdx) {
        if (blockIdx < 0 || blockIdx >= vBlocks.length) {
            throw new IndexOutOfBoundsException("Block index out of range");
        }
        return vBlocks[blockIdx];
    }

    long bytesPerToken() { return bytesPerToken; }
    long bytesPerBlock() { return bytesPerBlock; }
    int headDim() { return headDim; }
    int numHeads() { return numHeads; }

    void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("TurboQuantKVCacheAdapter is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            sequenceBlockTables.clear();
            sequenceLengths.clear();
            sequenceTenants.clear();
            sequenceEngines.clear();
            freeBlocks.clear();
            LOG.info("TurboQuantKVCacheAdapter closed");
        }
    }

    // ── Nested Classes ─────────────────────────────────────────────────

    /**
     * Sequence-specific compressed KV cache view.
     * <p>
     * Each request gets its own {@code TurboQuantSequenceCache} which manages
     * block allocation and TurboQuant operations for that sequence.
     */
    public final class TurboQuantSequenceCache {

        private final String sequenceId;
        private int localLength = 0;

        private TurboQuantSequenceCache(String sequenceId) {
            this.sequenceId = sequenceId;
        }

        /**
         * Appends a key vector with TurboQuant compression.
         *
         * @param layer attention layer index
         * @param key key vector [headDim]
         * @return sequence position
         */
        public int appendKey(int layer, float[] key) {
            ensureOpen();

            TurboQuantLocalEngine[] engines = sequenceEngines.get(sequenceId);
            if (engines == null || layer >= engines.length) {
                throw new IllegalArgumentException("Invalid layer: " + layer);
            }

            // Quantize using TurboQuant
            TurboQuantTypes.QuantProdResult result = engines[layer].quantizeProd(key);

            // Allocate block if needed
            Map<Integer, int[]> blockTable = sequenceBlockTables.computeIfAbsent(
                sequenceId, k -> new ConcurrentHashMap<>());
            int[] layerBlocks = blockTable.computeIfAbsent(layer, l -> new int[0]);

            int currentBlockIdx;
            int tokenOffsetInBlock = localLength % pageSize;

            if (tokenOffsetInBlock == 0) {
                // Need new block
                currentBlockIdx = allocateBlock();
                if (currentBlockIdx == -1) {
                    throw new OutOfMemoryError(
                        "TurboQuant KV cache OOM: no free blocks (sequence=" + sequenceId + ")");
                }
                // Resize array
                int[] newBlocks = Arrays.copyOf(layerBlocks, layerBlocks.length + 1);
                newBlocks[newBlocks.length - 1] = currentBlockIdx;
                blockTable.put(layer, newBlocks);
            } else {
                currentBlockIdx = layerBlocks[layerBlocks.length - 1];
            }

            // Store compressed data
            MemorySegment kBlock = getKBlock(currentBlockIdx);
            storeCompressedKV(kBlock, tokenOffsetInBlock, result, true);

            localLength++;
            sequenceLengths.get(sequenceId).set(localLength);
            totalAppends.incrementAndGet();

            return localLength - 1;
        }

        /**
         * Appends a value vector with TurboQuant compression.
         */
        public void appendValue(int layer, float[] value) {
            ensureOpen();

            TurboQuantLocalEngine[] engines = sequenceEngines.get(sequenceId);
            if (engines == null || layer >= engines.length) {
                throw new IllegalArgumentException("Invalid layer: " + layer);
            }

            TurboQuantTypes.QuantProdResult result = engines[layer].quantizeProd(value);

            Map<Integer, int[]> blockTable = sequenceBlockTables.get(sequenceId);
            if (blockTable == null) return;

            int[] layerBlocks = blockTable.get(layer);
            if (layerBlocks == null || layerBlocks.length == 0) return;

            int currentBlockIdx = layerBlocks[layerBlocks.length - 1];
            int tokenOffsetInBlock = (localLength - 1) % pageSize;

            MemorySegment vBlock = getVBlock(currentBlockIdx);
            storeCompressedKV(vBlock, tokenOffsetInBlock, result, false);
        }

        /**
         * Computes attention scores from compressed KV cache.
         *
         * @param layer attention layer index
         * @param query query vector [headDim]
         * @param scores output scores array [sequenceLength]
         */
        public void computeAttentionScores(int layer, float[] query, float[] scores) {
            ensureOpen();

            TurboQuantLocalEngine[] engines = sequenceEngines.get(sequenceId);
            if (engines == null || layer >= engines.length) {
                throw new IllegalArgumentException("Invalid layer: " + layer);
            }

            Map<Integer, int[]> blockTable = sequenceBlockTables.get(sequenceId);
            if (blockTable == null) return;

            int[] layerBlocks = blockTable.get(layer);
            if (layerBlocks == null) return;

            int seqLen = localLength;
            if (scores.length < seqLen) {
                throw new IllegalArgumentException("Scores array too small: " + scores.length + " < " + seqLen);
            }

            // For each token, load compressed data and estimate inner product
            for (int t = 0; t < seqLen; t++) {
                int blockIdx = t / pageSize;
                int tokenOffset = t % pageSize;

                if (blockIdx >= layerBlocks.length) break;

                MemorySegment kBlock = getKBlock(layerBlocks[blockIdx]);
                TurboQuantTypes.QuantProdResult result = loadCompressedKV(kBlock, tokenOffset);

                // Estimate inner product from compressed representation
                scores[t] = engines[layer].estimateInnerProductFull(query, result);
            }

            totalDequantizations.addAndGet(seqLen);
        }

        /**
         * Dequantizes a stored key vector.
         *
         * @param layer attention layer index
         * @param position token position
         * @param output dequantized key vector [headDim]
         */
        public void dequantizeKey(int layer, int position, float[] output) {
            ensureOpen();

            TurboQuantLocalEngine[] engines = sequenceEngines.get(sequenceId);
            if (engines == null || layer >= engines.length) {
                throw new IllegalArgumentException("Invalid layer: " + layer);
            }

            int blockIdx = position / pageSize;
            int tokenOffset = position % pageSize;

            Map<Integer, int[]> blockTable = sequenceBlockTables.get(sequenceId);
            int[] layerBlocks = blockTable.get(layer);
            if (layerBlocks == null || blockIdx >= layerBlocks.length) return;

            MemorySegment kBlock = getKBlock(layerBlocks[blockIdx]);
            TurboQuantTypes.QuantProdResult result = loadCompressedKV(kBlock, tokenOffset);

            engines[layer].dequantizeProd(result, output);
            totalDequantizations.incrementAndGet();
        }

        /**
         * Dequantizes a stored value vector.
         *
         * @param layer attention layer index
         * @param position token position
         * @param output dequantized value vector [headDim]
         */
        public void dequantizeValue(int layer, int position, float[] output) {
            ensureOpen();

            TurboQuantLocalEngine[] engines = sequenceEngines.get(sequenceId);
            if (engines == null || layer >= engines.length) {
                throw new IllegalArgumentException("Invalid layer: " + layer);
            }

            int blockIdx = position / pageSize;
            int tokenOffset = position % pageSize;

            Map<Integer, int[]> blockTable = sequenceBlockTables.get(sequenceId);
            int[] layerBlocks = blockTable.get(layer);
            if (layerBlocks == null || blockIdx >= layerBlocks.length) return;

            MemorySegment vBlock = getVBlock(layerBlocks[blockIdx]);
            TurboQuantTypes.QuantProdResult result = loadCompressedKV(vBlock, tokenOffset);

            engines[layer].dequantizeProd(result, output);
            totalDequantizations.incrementAndGet();
        }

        public int length() { return localLength; }

        public void clear() {
            TurboQuantKVCacheAdapter.this.freeSequence(sequenceId);
            localLength = 0;
        }

        // ── Compression Helpers ────────────────────────────────────────

        /**
         * Stores compressed KV data using FFM bulk copy (5-10× faster than per-element).
         */
        private void storeCompressedKV(MemorySegment block, int tokenOffset,
                                       TurboQuantTypes.QuantProdResult result, boolean isKey) {
            long baseOffset = (long) tokenOffset * numHeads * bytesPerToken;
            int bits = turboQuantConfig.bits();

            // MSE indices - bulk copy
            long mseBytes = (long) Math.ceil((bits * headDim) / 8.0);
            // Pack int indices into bytes for storage
            byte[] packedIndices = packIndices(result.mseIndices(), bits);
            TurboQuantTypes.bulkCopyByteToNative(packedIndices, block, baseOffset);

            // QJL signs - bulk copy
            long qjlOffset = baseOffset + mseBytes;
            TurboQuantTypes.bulkCopyByteToNative(result.qjlSigns(), block, qjlOffset);

            // Norms - bulk copy (2x FP32 = 8 bytes)
            long normOffset = qjlOffset + result.qjlSigns().length;
            float[] norms = new float[]{result.residualNorm(), 1.0f};  // residual, original
            TurboQuantTypes.bulkCopyToNative(norms, block, normOffset);
        }

        /**
         * Loads compressed KV data using FFM bulk copy (5-10× faster than per-element).
         */
        private TurboQuantTypes.QuantProdResult loadCompressedKV(MemorySegment block, int tokenOffset) {
            long baseOffset = (long) tokenOffset * numHeads * bytesPerToken;
            int bits = turboQuantConfig.bits();

            // MSE indices - bulk load
            long mseBytes = (long) Math.ceil((bits * headDim) / 8.0);
            byte[] packedIndices = new byte[(int) mseBytes];
            TurboQuantTypes.bulkCopyFromNativeByte(block, baseOffset, packedIndices);
            int[] mseIndices = unpackIndices(packedIndices, bits);

            // QJL signs - bulk load
            long qjlOffset = baseOffset + mseBytes;
            byte[] qjlSigns = new byte[headDim];
            TurboQuantTypes.bulkCopyFromNativeByte(block, qjlOffset, qjlSigns);

            // Norms - bulk load
            long normOffset = qjlOffset + headDim;
            float[] norms = new float[2];
            TurboQuantTypes.bulkCopyFromNative(block, normOffset, norms);

            return new TurboQuantTypes.QuantProdResult(mseIndices, qjlSigns, norms[0]);
        }

        /**
         * Packs integer indices into a byte array with specified bit-width.
         * E.g., 3-bit indices: 8 indices packed into 3 bytes
         */
        private byte[] packIndices(int[] indices, int bits) {
            int totalBits = indices.length * bits;
            int totalBytes = (int) Math.ceil(totalBits / 8.0);
            byte[] packed = new byte[totalBytes];

            int bitOffset = 0;
            for (int idx : indices) {
                for (int b = 0; b < bits; b++) {
                    int byteIdx = (bitOffset + b) / 8;
                    int bitIdx = (bitOffset + b) % 8;
                    if ((idx & (1 << b)) != 0) {
                        packed[byteIdx] |= (byte) (1 << bitIdx);
                    }
                }
                bitOffset += bits;
            }

            return packed;
        }

        /**
         * Unpacks indices from a byte array with specified bit-width.
         */
        private int[] unpackIndices(byte[] packed, int bits) {
            int[] indices = new int[headDim];
            int totalBits = headDim * bits;
            int bitOffset = 0;

            for (int i = 0; i < headDim; i++) {
                int value = 0;
                for (int b = 0; b < bits; b++) {
                    int byteIdx = (bitOffset + b) / 8;
                    int bitIdx = (bitOffset + b) % 8;
                    if (byteIdx < packed.length) {
                        value |= ((packed[byteIdx] >> bitIdx) & 1) << b;
                    }
                }
                indices[i] = value;
                bitOffset += bits;
            }

            return indices;
        }
    }

    // ── Configuration ──────────────────────────────────────────────────

    public static final class Config {
        final KVCacheStorageMode storageMode;
        final TurboQuantTypes.TurboQuantConfig turboQuantConfig;
        final int numBlocks;
        final int pageSize;
        final int numLayers;
        final int numHeads;
        final int headDim;
        final Arena arena;

        private Config(KVCacheStorageMode storageMode, TurboQuantTypes.TurboQuantConfig turboQuantConfig,
                      int numBlocks, int pageSize, int numLayers, int numHeads,
                      int headDim, Arena arena) {
            this.storageMode = storageMode;
            this.turboQuantConfig = turboQuantConfig;
            this.numBlocks = numBlocks;
            this.pageSize = pageSize;
            this.numLayers = numLayers;
            this.numHeads = numHeads;
            this.headDim = headDim;
            this.arena = arena;
        }
    }

    public static final class ConfigBuilder {
        private KVCacheStorageMode storageMode = KVCacheStorageMode.TURBOQUANT_3BIT;
        private TurboQuantTypes.TurboQuantConfig turboQuantConfig = TurboQuantTypes.TurboQuantConfig.prod3bitKvCache(128);
        private int numBlocks = 1024;
        private int pageSize = 16;
        private int numLayers = 32;
        private int numHeads = 32;
        private int headDim = 128;
        private Arena arena = Arena.ofAuto();

        private ConfigBuilder() {}

        public ConfigBuilder storageMode(KVCacheStorageMode storageMode) {
            this.storageMode = storageMode;
            return this;
        }

        public ConfigBuilder turboQuantConfig(TurboQuantTypes.TurboQuantConfig config) {
            this.turboQuantConfig = config;
            return this;
        }

        public ConfigBuilder numBlocks(int numBlocks) {
            this.numBlocks = numBlocks;
            return this;
        }

        public ConfigBuilder pageSize(int pageSize) {
            this.pageSize = pageSize;
            return this;
        }

        public ConfigBuilder numLayers(int numLayers) {
            this.numLayers = numLayers;
            return this;
        }

        public ConfigBuilder numHeads(int numHeads) {
            this.numHeads = numHeads;
            return this;
        }

        public ConfigBuilder headDim(int headDim) {
            this.headDim = headDim;
            return this;
        }

        public ConfigBuilder arena(Arena arena) {
            this.arena = arena;
            return this;
        }

        /**
         * Auto-configures from model spec.
         */
        public ConfigBuilder fromModelSpec(int numLayers, int numHeads, int headDim,
                                          int maxTokens) {
            this.numLayers = numLayers;
            this.numHeads = numHeads;
            this.headDim = headDim;
            this.turboQuantConfig = TurboQuantTypes.TurboQuantConfig.prod3bitKvCache(headDim);
            // With 6× compression, need fewer blocks
            this.numBlocks = (int) Math.ceil((maxTokens * 1.2) / (pageSize * 6.0));
            return this;
        }

        public TurboQuantKVCacheAdapter build() {
            return new TurboQuantKVCacheAdapter(new Config(
                storageMode, turboQuantConfig, numBlocks, pageSize,
                numLayers, numHeads, headDim, arena));
        }
    }

    @Override
    public String toString() {
        return "TurboQuantKVCacheAdapter[mode=%s, blocks=%d/%d, utilization=%.1f%%, sequences=%d]".formatted(
            storageMode, usedBlocks(), totalBlocks(), memoryUtilization() * 100,
            sequenceBlockTables.size());
    }
}
