/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.generation;

import java.util.Collections;
import java.util.List;

/**
 * Immutable sampling configuration for an autoregressive generation session.
 */
public final class GenerationConfig {

    public enum SamplingStrategy {
        GREEDY, TOP_K, TOP_P, TOP_K_TOP_P, BEAM
    }

    private final SamplingStrategy strategy;
    private final float temperature;
    private final int topK;
    private final float topP;
    private final int beamWidth;
    private final int maxNewTokens;
    private final int minNewTokens;
    private final List<Integer> stopTokenIds;
    private final List<String> stopStrings;
    private final float repetitionPenalty;
    private final float frequencyPenalty;
    private final boolean useKvCache;
    private final int maxKvCacheTokens;
    private final long seed;

    private GenerationConfig(Builder b) {
        this.strategy = b.strategy;
        this.temperature = b.temperature;
        this.topK = b.topK;
        this.topP = b.topP;
        this.beamWidth = b.beamWidth;
        this.maxNewTokens = b.maxNewTokens;
        this.minNewTokens = b.minNewTokens;
        this.stopTokenIds = List.copyOf(b.stopTokenIds);
        this.stopStrings = List.copyOf(b.stopStrings);
        this.repetitionPenalty = b.repetitionPenalty;
        this.frequencyPenalty = b.frequencyPenalty;
        this.useKvCache = b.useKvCache;
        this.maxKvCacheTokens = b.maxKvCacheTokens;
        this.seed = b.seed;
    }

    /**
     * Returns a default greedy config with {@code maxNewTokens=512}.
     *
     * @return a greedy {@link GenerationConfig}
     */
    public static GenerationConfig defaults() {
        return builder().strategy(SamplingStrategy.GREEDY).maxNewTokens(512).build();
    }

    /**
     * Creates a new builder with all defaults applied.
     *
     * @return a new {@link Builder}
     */
    public static Builder builder() {
        return new Builder();
    }

    /** @return the sampling strategy */
    public SamplingStrategy strategy() { return strategy; }
    /** @return sampling temperature (default: 1.0) */
    public float temperature() { return temperature; }
    /** @return top-k candidate count (default: 50) */
    public int topK() { return topK; }
    /** @return nucleus sampling threshold (default: 1.0 = disabled) */
    public float topP() { return topP; }
    /** @return beam width for beam search (default: 1) */
    public int beamWidth() { return beamWidth; }
    /** @return maximum number of new tokens to generate (default: 512) */
    public int maxNewTokens() { return maxNewTokens; }
    /** @return minimum number of new tokens to generate (default: 1) */
    public int minNewTokens() { return minNewTokens; }
    /** @return token IDs that terminate generation early */
    public List<Integer> stopTokenIds() { return stopTokenIds; }
    /** @return stop strings that terminate generation early */
    public List<String> stopStrings() { return stopStrings; }
    /** @return repetition penalty applied to already-generated tokens (default: 1.0 = none) */
    public float repetitionPenalty() { return repetitionPenalty; }
    /** @return frequency penalty applied to token logits (default: 0.0 = none) */
    public float frequencyPenalty() { return frequencyPenalty; }
    /** @return {@code true} if KV caching is enabled (default: true) */
    public boolean useKvCache() { return useKvCache; }
    /** @return maximum number of tokens the KV cache can hold (default: 8192) */
    public int maxKvCacheTokens() { return maxKvCacheTokens; }
    /** @return RNG seed; {@code -1} means random (default: -1) */
    public long seed() { return seed; }

    /**
     * Returns {@code true} if the sampling strategy is {@link SamplingStrategy#GREEDY}.
     *
     * @return {@code true} for greedy decoding
     */
    public boolean isGreedy() {
        return strategy == SamplingStrategy.GREEDY;
    }

    /**
     * Builder for {@link GenerationConfig}.
     */
    public static final class Builder {
        private SamplingStrategy strategy = SamplingStrategy.GREEDY;
        private float temperature = 1.0f;
        private int topK = 50;
        private float topP = 1.0f;
        private int beamWidth = 1;
        private int maxNewTokens = 512;
        private int minNewTokens = 1;
        private List<Integer> stopTokenIds = Collections.emptyList();
        private List<String> stopStrings = Collections.emptyList();
        private float repetitionPenalty = 1.0f;
        private float frequencyPenalty = 0.0f;
        private boolean useKvCache = true;
        private int maxKvCacheTokens = 8192;
        private long seed = -1L;

        /** @param v sampling strategy */
        public Builder strategy(SamplingStrategy v) { this.strategy = v; return this; }
        /** @param v sampling temperature; lower = more deterministic */
        public Builder temperature(float v) { this.temperature = v; return this; }
        /** @param v top-k candidate count */
        public Builder topK(int v) { this.topK = v; return this; }
        /** @param v nucleus sampling threshold in {@code (0, 1]}; 1.0 disables */
        public Builder topP(float v) { this.topP = v; return this; }
        /** @param v beam width for beam search; 1 = greedy */
        public Builder beamWidth(int v) { this.beamWidth = v; return this; }
        /** @param v maximum new tokens to generate; must be &gt; 0 */
        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        /** @param v minimum new tokens before stop sequences are checked */
        public Builder minNewTokens(int v) { this.minNewTokens = v; return this; }
        /** @param v token IDs that terminate generation */
        public Builder stopTokenIds(List<Integer> v) { this.stopTokenIds = v; return this; }
        /** @param v strings that terminate generation */
        public Builder stopStrings(List<String> v) { this.stopStrings = v; return this; }
        /** @param v repetition penalty; 1.0 = no penalty */
        public Builder repetitionPenalty(float v) { this.repetitionPenalty = v; return this; }
        /** @param v frequency penalty; 0.0 = no penalty */
        public Builder frequencyPenalty(float v) { this.frequencyPenalty = v; return this; }
        /** @param v {@code true} to enable KV caching */
        public Builder useKvCache(boolean v) { this.useKvCache = v; return this; }
        /** @param v maximum KV cache capacity in tokens */
        public Builder maxKvCacheTokens(int v) { this.maxKvCacheTokens = v; return this; }
        /** @param v RNG seed; {@code -1} for random */
        public Builder seed(long v) { this.seed = v; return this; }

        /**
         * Builds the {@link GenerationConfig}.
         *
         * @return a new immutable {@link GenerationConfig}
         */
        public GenerationConfig build() {
            return new GenerationConfig(this);
        }
    }
}
