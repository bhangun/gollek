
import java.time.Duration;
import java.util.Objects;

/**
 * Configuration for GGUF model loading and inference
 */
public class GGUFConfig {

    private final int nGpuLayers;
    private final int nThreads;
    private final int nCtx;
    private final int nBatch;
    private final boolean useMmap;
    private final boolean useMlock;
    private final int seed;
    private final float temperature;
    private final float topP;
    private final int topK;
    private final float repeatPenalty;
    private final int repeatLastN;
    private final Duration timeout;

    private GGUFConfig(Builder builder) {
        this.nGpuLayers = builder.nGpuLayers;
        this.nThreads = builder.nThreads;
        this.nCtx = builder.nCtx;
        this.nBatch = builder.nBatch;
        this.useMmap = builder.useMmap;
        this.useMlock = builder.useMlock;
        this.seed = builder.seed;
        this.temperature = builder.temperature;
        this.topP = builder.topP;
        this.topK = builder.topK;
        this.repeatPenalty = builder.repeatPenalty;
        this.repeatLastN = builder.repeatLastN;
        this.timeout = builder.timeout;
    }

    public static Builder builder() {
        return new Builder();
    }

    // Getters
    public int getNGpuLayers() {
        return nGpuLayers;
    }

    public int getNThreads() {
        return nThreads;
    }

    public int getNCtx() {
        return nCtx;
    }

    public int getNBatch() {
        return nBatch;
    }

    public boolean isUseMmap() {
        return useMmap;
    }

    public boolean isUseMlock() {
        return useMlock;
    }

    public int getSeed() {
        return seed;
    }

    public float getTemperature() {
        return temperature;
    }

    public float getTopP() {
        return topP;
    }

    public int getTopK() {
        return topK;
    }

    public float getRepeatPenalty() {
        return repeatPenalty;
    }

    public int getRepeatLastN() {
        return repeatLastN;
    }

    public Duration getTimeout() {
        return timeout;
    }

    public static class Builder {
        private int nGpuLayers = 0; // 0 = CPU only
        private int nThreads = Runtime.getRuntime().availableProcessors();
        private int nCtx = 2048;
        private int nBatch = 512;
        private boolean useMmap = true;
        private boolean useMlock = false;
        private int seed = -1; // -1 = random
        private float temperature = 0.8f;
        private float topP = 0.95f;
        private int topK = 40;
        private float repeatPenalty = 1.1f;
        private int repeatLastN = 64;
        private Duration timeout = Duration.ofSeconds(30);

        public Builder nGpuLayers(int nGpuLayers) {
            this.nGpuLayers = nGpuLayers;
            return this;
        }

        public Builder nThreads(int nThreads) {
            this.nThreads = nThreads;
            return this;
        }

        public Builder nCtx(int nCtx) {
            this.nCtx = nCtx;
            return this;
        }

        public Builder nBatch(int nBatch) {
            this.nBatch = nBatch;
            return this;
        }

        public Builder useMmap(boolean useMmap) {
            this.useMmap = useMmap;
            return this;
        }

        public Builder useMlock(boolean useMlock) {
            this.useMlock = useMlock;
            return this;
        }

        public Builder seed(int seed) {
            this.seed = seed;
            return this;
        }

        public Builder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topP(float topP) {
            this.topP = topP;
            return this;
        }

        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }

        public Builder repeatPenalty(float repeatPenalty) {
            this.repeatPenalty = repeatPenalty;
            return this;
        }

        public Builder repeatLastN(int repeatLastN) {
            this.repeatLastN = repeatLastN;
            return this;
        }

        public GGUFConfig build() {
            return new GGUFConfig(this);
        }
    }
}