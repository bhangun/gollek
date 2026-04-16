package tech.kayys.gollek.inference.gguf;

import java.time.Duration;
import java.util.Collections;
import java.util.List;

/**
 * Immutable sampling and generation parameters for a single GGUF inference call.
 *
 * <p>Covers the full set of llama.cpp sampling knobs: temperature, top-k/p,
 * repeat penalties, Mirostat, grammar constraints, and JSON mode.
 * Use {@link #builder()} to construct instances.
 *
 * <pre>{@code
 * GenerationParams params = GenerationParams.builder()
 *     .maxTokens(256)
 *     .temperature(0.7f)
 *     .topP(0.9f)
 *     .stopTokens(List.of("<|im_end|>"))
 *     .build();
 * }</pre>
 *
 * @see GGUFProvider
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

    /**
     * Creates a new builder with all defaults applied.
     *
     * @return a new {@link Builder}
     */
    public static Builder builder() {
        return new Builder();
    }

    /** @return maximum number of tokens to generate (default: 512) */
    public int getMaxTokens() { return maxTokens; }

    /** @return sampling temperature; lower = more deterministic (default: 0.8) */
    public float getTemperature() { return temperature; }

    /** @return nucleus sampling threshold (default: 0.95) */
    public float getTopP() { return topP; }

    /** @return top-k candidate count (default: 40) */
    public int getTopK() { return topK; }

    /** @return repetition penalty applied to already-generated tokens (default: 1.1) */
    public float getRepeatPenalty() { return repeatPenalty; }

    /** @return number of last tokens considered for repeat penalty (default: 64) */
    public int getRepeatLastN() { return repeatLastN; }

    /** @return presence penalty to reduce token repetition (default: 0.0) */
    public float getPresencePenalty() { return presencePenalty; }

    /** @return frequency penalty to reduce token frequency (default: 0.0) */
    public float getFrequencyPenalty() { return frequencyPenalty; }

    /** @return Mirostat target entropy τ (default: 5.0) */
    public float getMirostatTau() { return mirostatTau; }

    /** @return Mirostat learning rate η (default: 0.1) */
    public float getMirostatEta() { return mirostatEta; }

    /** @return Mirostat mode: 0 = disabled, 1 = Mirostat v1, 2 = Mirostat v2 (default: 0) */
    public int getMirostatMode() { return mirostatMode; }

    /** @return GBNF grammar string for constrained generation, or {@code null} if unconstrained */
    public String getGrammar() { return grammar; }

    /** @return {@code true} if JSON-only output mode is enabled */
    public boolean isJsonMode() { return jsonMode; }

    /** @return stop sequences that terminate generation early */
    public List<String> getStopTokens() { return stopTokens; }

    /** @return {@code true} if tokens should be streamed incrementally */
    public boolean isStream() { return stream; }

    /**
     * Builder for {@link GenerationParams}.
     */
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
        private int mirostatMode = 0;
        private String grammar = null;
        private boolean jsonMode = true;
        private List<String> stopTokens = Collections.emptyList();
        private boolean stream = false;
        private Duration timeout = Duration.ofSeconds(30);

        /** @param maxTokens maximum tokens to generate; must be &gt; 0 */
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }

        /** @param temperature sampling temperature in range {@code [0.0, 2.0]} */
        public Builder temperature(float temperature) { this.temperature = temperature; return this; }

        /** @param topP nucleus sampling threshold in range {@code (0.0, 1.0]} */
        public Builder topP(float topP) { this.topP = topP; return this; }

        /** @param topK top-k candidate count */
        public Builder topK(int topK) { this.topK = topK; return this; }

        /** @param repeatPenalty penalty for repeating tokens; 1.0 = no penalty */
        public Builder repeatPenalty(float repeatPenalty) { this.repeatPenalty = repeatPenalty; return this; }

        /** @param repeatLastN number of last tokens to consider for repeat penalty */
        public Builder repeatLastN(int repeatLastN) { this.repeatLastN = repeatLastN; return this; }

        /** @param presencePenalty presence penalty; 0.0 = no penalty */
        public Builder presencePenalty(float presencePenalty) { this.presencePenalty = presencePenalty; return this; }

        /** @param frequencyPenalty frequency penalty; 0.0 = no penalty */
        public Builder frequencyPenalty(float frequencyPenalty) { this.frequencyPenalty = frequencyPenalty; return this; }

        /** @param mirostatTau Mirostat target entropy τ */
        public Builder mirostatTau(float mirostatTau) { this.mirostatTau = mirostatTau; return this; }

        /** @param mirostatEta Mirostat learning rate η */
        public Builder mirostatEta(float mirostatEta) { this.mirostatEta = mirostatEta; return this; }

        /** @param mirostatMode 0 = disabled, 1 = v1, 2 = v2 */
        public Builder mirostatMode(int mirostatMode) { this.mirostatMode = mirostatMode; return this; }

        /** @param grammar GBNF grammar string; {@code null} = unconstrained */
        public Builder grammar(String grammar) { this.grammar = grammar; return this; }

        /** @param jsonMode {@code true} to constrain output to valid JSON */
        public Builder jsonMode(boolean jsonMode) { this.jsonMode = jsonMode; return this; }

        /** @param stopTokens strings that terminate generation; {@code null} treated as empty */
        public Builder stopTokens(List<String> stopTokens) {
            this.stopTokens = stopTokens != null ? stopTokens : Collections.emptyList();
            return this;
        }

        /** @param stream {@code true} to stream tokens incrementally */
        public Builder stream(boolean stream) { this.stream = stream; return this; }

        /**
         * Builds the {@link GenerationParams}.
         *
         * @return a new immutable {@link GenerationParams}
         */
        public GenerationParams build() { return new GenerationParams(this); }
    }
}