package tech.kayys.gollek.inference.libtorch;

import java.util.Collections;
import java.util.List;

/**
 * Generation parameters for LibTorch text generation.
 * <p>
 * Mirrors the GGUF {@code GenerationParams} builder pattern,
 * providing a unified configuration for sampling strategies
 * and generation control.
 */
public class LibTorchGenerationParams {

    private final int maxTokens;
    private final float temperature;
    private final float topP;
    private final int topK;
    private final float repeatPenalty;
    private final int repeatLastN;
    private final float presencePenalty;
    private final float frequencyPenalty;
    private final List<String> stopTokens;
    private final boolean stream;
    private final String samplingStrategy;

    private LibTorchGenerationParams(Builder builder) {
        this.maxTokens = builder.maxTokens;
        this.temperature = builder.temperature;
        this.topP = builder.topP;
        this.topK = builder.topK;
        this.repeatPenalty = builder.repeatPenalty;
        this.repeatLastN = builder.repeatLastN;
        this.presencePenalty = builder.presencePenalty;
        this.frequencyPenalty = builder.frequencyPenalty;
        this.stopTokens = builder.stopTokens;
        this.stream = builder.stream;
        this.samplingStrategy = builder.samplingStrategy;
    }

    public static Builder builder() {
        return new Builder();
    }

    public int getMaxTokens() {
        return maxTokens;
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

    public float getPresencePenalty() {
        return presencePenalty;
    }

    public float getFrequencyPenalty() {
        return frequencyPenalty;
    }

    public List<String> getStopTokens() {
        return stopTokens;
    }

    public boolean isStream() {
        return stream;
    }

    public String getSamplingStrategy() {
        return samplingStrategy;
    }

    @Override
    public String toString() {
        return String.format(
                "LibTorchGenerationParams{maxTokens=%d, temp=%.2f, topP=%.2f, topK=%d, strategy=%s}",
                maxTokens, temperature, topP, topK, samplingStrategy);
    }

    public static class Builder {
        private int maxTokens = 512;
        private float temperature = 0.8f;
        private float topP = 0.95f;
        private int topK = 40;
        private float repeatPenalty = 1.1f;
        private int repeatLastN = 64;
        private float presencePenalty = 0.0f;
        private float frequencyPenalty = 0.0f;
        private List<String> stopTokens = Collections.emptyList();
        private boolean stream = false;
        private String samplingStrategy = "top_p";

        public Builder maxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
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

        public Builder presencePenalty(float presencePenalty) {
            this.presencePenalty = presencePenalty;
            return this;
        }

        public Builder frequencyPenalty(float frequencyPenalty) {
            this.frequencyPenalty = frequencyPenalty;
            return this;
        }

        public Builder stopTokens(List<String> stopTokens) {
            this.stopTokens = stopTokens != null ? stopTokens : Collections.emptyList();
            return this;
        }

        public Builder stream(boolean stream) {
            this.stream = stream;
            return this;
        }

        public Builder samplingStrategy(String samplingStrategy) {
            this.samplingStrategy = samplingStrategy;
            return this;
        }

        public LibTorchGenerationParams build() {
            return new LibTorchGenerationParams(this);
        }
    }
}
