package tech.kayys.gollek.inference.gguf;

import java.time.Duration;
import java.util.Collections;
import java.util.List;

/**
 * Generation parameters for text generation
 */
public class GenerationParams {
    private final int maxTokens;
    private final float temperature;
    private final float topP;
    private final int topK;
    private final float repeatPenalty;
    private final int repeatLastN;
    private final float presencePenalty;
    private final float frequencyPenalty;
    private final float mirostatTau;
    private final float mirostatEta;
    private final int mirostatMode;
    private final String grammar;
    private final boolean jsonMode;
    private final List<String> stopTokens;
    private final boolean stream;

    private GenerationParams(Builder builder) {
        this.maxTokens = builder.maxTokens;
        this.temperature = builder.temperature;
        this.topP = builder.topP;
        this.topK = builder.topK;
        this.repeatPenalty = builder.repeatPenalty;
        this.repeatLastN = builder.repeatLastN;
        this.presencePenalty = builder.presencePenalty;
        this.frequencyPenalty = builder.frequencyPenalty;
        this.mirostatTau = builder.mirostatTau;
        this.mirostatEta = builder.mirostatEta;
        this.mirostatMode = builder.mirostatMode;
        this.grammar = builder.grammar;
        this.jsonMode = builder.jsonMode;
        this.stopTokens = builder.stopTokens;
        this.stream = builder.stream;
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

    public float getMirostatTau() {
        return mirostatTau;
    }

    public float getMirostatEta() {
        return mirostatEta;
    }

    public int getMirostatMode() {
        return mirostatMode;
    }

    public String getGrammar() {
        return grammar;
    }

    public boolean isJsonMode() {
        return jsonMode;
    }

    public List<String> getStopTokens() {
        return stopTokens;
    }

    public boolean isStream() {
        return stream;
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
        private float mirostatTau = 5.0f;
        private float mirostatEta = 0.1f;
        private int mirostatMode = 0; // 0 = disabled
        private String grammar = null;
        private boolean jsonMode = true;
        private List<String> stopTokens = Collections.emptyList();
        private boolean stream = false;
        private Duration timeout = Duration.ofSeconds(30);

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

        public Builder mirostatTau(float mirostatTau) {
            this.mirostatTau = mirostatTau;
            return this;
        }

        public Builder mirostatEta(float mirostatEta) {
            this.mirostatEta = mirostatEta;
            return this;
        }

        public Builder mirostatMode(int mirostatMode) {
            this.mirostatMode = mirostatMode;
            return this;
        }

        public Builder grammar(String grammar) {
            this.grammar = grammar;
            return this;
        }

        public Builder jsonMode(boolean jsonMode) {
            this.jsonMode = jsonMode;
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

        public GenerationParams build() {
            return new GenerationParams(this);
        }
    }
}