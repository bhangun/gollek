package tech.kayys.gollek.litert.config;

import java.util.Objects;

/**
 * LiteRT configuration.
 */
public class LiteRTConfig {

    /**
     * Number of CPU threads (0 = auto).
     */
    private int numThreads = 4;

    /**
     * Delegate preference: NONE, CPU, GPU, NNAPI, COREML, AUTO.
     */
    private Delegate delegate = Delegate.AUTO;

    /**
     * Enable XNNPACK optimization.
     */
    private boolean enableXnnpack = true;

    /**
     * Use memory pool for faster allocations.
     */
    private boolean useMemoryPool = true;

    /**
     * Memory pool size in bytes (0 = default 16MB).
     */
    private long poolSizeBytes = 0;

    /**
     * Model cache directory (optional).
     */
    private String cacheDir;

    public LiteRTConfig() {
    }

    public LiteRTConfig(int numThreads, Delegate delegate, boolean enableXnnpack, boolean useMemoryPool,
            long poolSizeBytes, String cacheDir) {
        this.numThreads = numThreads;
        this.delegate = delegate;
        this.enableXnnpack = enableXnnpack;
        this.useMemoryPool = useMemoryPool;
        this.poolSizeBytes = poolSizeBytes;
        this.cacheDir = cacheDir;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

    public Delegate getDelegate() {
        return delegate;
    }

    public void setDelegate(Delegate delegate) {
        this.delegate = delegate;
    }

    public boolean isEnableXnnpack() {
        return enableXnnpack;
    }

    public void setEnableXnnpack(boolean enableXnnpack) {
        this.enableXnnpack = enableXnnpack;
    }

    public boolean isUseMemoryPool() {
        return useMemoryPool;
    }

    public void setUseMemoryPool(boolean useMemoryPool) {
        this.useMemoryPool = useMemoryPool;
    }

    public long getPoolSizeBytes() {
        return poolSizeBytes;
    }

    public void setPoolSizeBytes(long poolSizeBytes) {
        this.poolSizeBytes = poolSizeBytes;
    }

    public String getCacheDir() {
        return cacheDir;
    }

    public void setCacheDir(String cacheDir) {
        this.cacheDir = cacheDir;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        LiteRTConfig that = (LiteRTConfig) o;
        return numThreads == that.numThreads && enableXnnpack == that.enableXnnpack
                && useMemoryPool == that.useMemoryPool && poolSizeBytes == that.poolSizeBytes
                && delegate == that.delegate && Objects.equals(cacheDir, that.cacheDir);
    }

    @Override
    public int hashCode() {
        return Objects.hash(numThreads, delegate, enableXnnpack, useMemoryPool, poolSizeBytes, cacheDir);
    }

    @Override
    public String toString() {
        return "LiteRTConfig{" +
                "numThreads=" + numThreads +
                ", delegate=" + delegate +
                ", enableXnnpack=" + enableXnnpack +
                ", useMemoryPool=" + useMemoryPool +
                ", poolSizeBytes=" + poolSizeBytes +
                ", cacheDir='" + cacheDir + '\'' +
                '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int numThreads = 4;
        private Delegate delegate = Delegate.AUTO;
        private boolean enableXnnpack = true;
        private boolean useMemoryPool = true;
        private long poolSizeBytes = 0;
        private String cacheDir;

        public Builder numThreads(int numThreads) {
            this.numThreads = numThreads;
            return this;
        }

        public Builder delegate(Delegate delegate) {
            this.delegate = delegate;
            return this;
        }

        public Builder enableXnnpack(boolean enableXnnpack) {
            this.enableXnnpack = enableXnnpack;
            return this;
        }

        public Builder useMemoryPool(boolean useMemoryPool) {
            this.useMemoryPool = useMemoryPool;
            return this;
        }

        public Builder poolSizeBytes(long poolSizeBytes) {
            this.poolSizeBytes = poolSizeBytes;
            return this;
        }

        public Builder cacheDir(String cacheDir) {
            this.cacheDir = cacheDir;
            return this;
        }

        public LiteRTConfig build() {
            return new LiteRTConfig(numThreads, delegate, enableXnnpack, useMemoryPool, poolSizeBytes, cacheDir);
        }
    }

    /**
     * Hardware delegate types.
     */
    public enum Delegate {
        NONE,
        CPU,
        GPU,
        NNAPI,
        COREML,
        AUTO
    }
}
