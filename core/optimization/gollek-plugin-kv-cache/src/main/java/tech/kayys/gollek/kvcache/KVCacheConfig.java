package tech.kayys.gollek.kvcache;

/**
 * Configuration for the Paged KV-Cache Manager.
 * <p>
 * Defines the physical memory layout: how many blocks to pre-allocate,
 * how many tokens each block holds, and the model dimensions.
 * <p>
 * These values must match the target model's architecture. For example,
 * Llama-3-8B uses 32 layers, 32 heads, and 128 head dimensions.
 */
public class KVCacheConfig {

    /** Number of tokens stored per physical block */
    private int blockSize = 16;

    /** Total number of physical blocks to pre-allocate in the pool */
    private int totalBlocks = 1024;

    /** Number of transformer layers in the model (e.g., 32 for Llama-3-8B) */
    private int numLayers = 32;

    /** Number of attention heads per layer (e.g., 32 for Llama-3-8B) */
    private int numHeads = 32;

    /** Dimension of each attention head (e.g., 128 for Llama-3-8B) */
    private int headDim = 128;

    /**
     * Whether to use GPU memory for the cache pool.
     * When false, uses CPU off-heap memory (useful for development/testing).
     */
    private boolean useGpu = false;

    /**
     * Maximum number of blocks a single request can hold.
     * Prevents a single long sequence from monopolizing the pool.
     * 0 means unlimited.
     */
    private int maxBlocksPerRequest = 0;

    // -- Constructors --

    public KVCacheConfig() {
    }

    public KVCacheConfig(int blockSize, int totalBlocks, int numLayers,
                         int numHeads, int headDim, boolean useGpu,
                         int maxBlocksPerRequest) {
        this.blockSize = blockSize;
        this.totalBlocks = totalBlocks;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.useGpu = useGpu;
        this.maxBlocksPerRequest = maxBlocksPerRequest;
    }

    // -- Derived calculations --

    /**
     * Bytes required for a single KV block.
     * Each block stores K and V values for each layer.
     * Formula: blockSize × numLayers × numHeads × headDim × 2 (K+V) × 2 (Float16 bytes)
     */
    public long bytesPerBlock() {
        return (long) blockSize * numLayers * numHeads * headDim * 2L * 2L;
    }

    /**
     * Total bytes required for the entire block pool.
     */
    public long totalPoolBytes() {
        return bytesPerBlock() * totalBlocks;
    }

    /**
     * Maximum context length supported by the pool.
     */
    public int maxContextLength() {
        return blockSize * totalBlocks;
    }

    // -- Getters/Setters --

    public int getBlockSize() {
        return blockSize;
    }

    public void setBlockSize(int blockSize) {
        this.blockSize = blockSize;
    }

    public int getTotalBlocks() {
        return totalBlocks;
    }

    public void setTotalBlocks(int totalBlocks) {
        this.totalBlocks = totalBlocks;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public void setNumHeads(int numHeads) {
        this.numHeads = numHeads;
    }

    public int getHeadDim() {
        return headDim;
    }

    public void setHeadDim(int headDim) {
        this.headDim = headDim;
    }

    public boolean isUseGpu() {
        return useGpu;
    }

    public void setUseGpu(boolean useGpu) {
        this.useGpu = useGpu;
    }

    public int getMaxBlocksPerRequest() {
        return maxBlocksPerRequest;
    }

    public void setMaxBlocksPerRequest(int maxBlocksPerRequest) {
        this.maxBlocksPerRequest = maxBlocksPerRequest;
    }

    @Override
    public String toString() {
        return "KVCacheConfig{" +
                "blockSize=" + blockSize +
                ", totalBlocks=" + totalBlocks +
                ", numLayers=" + numLayers +
                ", numHeads=" + numHeads +
                ", headDim=" + headDim +
                ", useGpu=" + useGpu +
                ", maxBlocksPerRequest=" + maxBlocksPerRequest +
                ", bytesPerBlock=" + bytesPerBlock() +
                ", totalPoolMB=" + (totalPoolBytes() / (1024 * 1024)) +
                '}';
    }

    // -- Builder --

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final KVCacheConfig config = new KVCacheConfig();

        public Builder blockSize(int blockSize) {
            config.setBlockSize(blockSize);
            return this;
        }

        public Builder totalBlocks(int totalBlocks) {
            config.setTotalBlocks(totalBlocks);
            return this;
        }

        public Builder numLayers(int numLayers) {
            config.setNumLayers(numLayers);
            return this;
        }

        public Builder numHeads(int numHeads) {
            config.setNumHeads(numHeads);
            return this;
        }

        public Builder headDim(int headDim) {
            config.setHeadDim(headDim);
            return this;
        }

        public Builder useGpu(boolean useGpu) {
            config.setUseGpu(useGpu);
            return this;
        }

        public Builder maxBlocksPerRequest(int max) {
            config.setMaxBlocksPerRequest(max);
            return this;
        }

        public KVCacheConfig build() {
            if (config.blockSize <= 0) throw new IllegalArgumentException("blockSize must be > 0");
            if (config.totalBlocks <= 0) throw new IllegalArgumentException("totalBlocks must be > 0");
            if (config.numLayers <= 0) throw new IllegalArgumentException("numLayers must be > 0");
            if (config.numHeads <= 0) throw new IllegalArgumentException("numHeads must be > 0");
            if (config.headDim <= 0) throw new IllegalArgumentException("headDim must be > 0");
            return config;
        }
    }
}
