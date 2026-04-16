/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.plugin.optimization;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Optimization configuration with model parameters and execution settings.
 *
 * @param modelPath Path to model file
 * @param numHeads Number of attention heads
 * @param numKVHeads Number of KV heads (for GQA)
 * @param headDim Head dimension
 * @param seqLen Sequence length
 * @param hiddenDim Hidden dimension
 * @param numLayers Number of transformer layers
 * @param metadata Additional metadata
 * @since 2.0.0
 */
public record OptimizationConfig(
    String modelPath,
    int numHeads,
    int numKVHeads,
    int headDim,
    int seqLen,
    int hiddenDim,
    int numLayers,
    Map<String, Object> metadata
) {
    /**
     * Create config with defaults.
     */
    public OptimizationConfig {
        Objects.requireNonNull(modelPath);
    }

    /**
     * Get metadata value.
     *
     * @param key metadata key
     * @return optional value
     */
    public Optional<Object> getMetadata(String key) {
        return Optional.ofNullable(metadata.get(key));
    }

    /**
     * Get typed metadata value.
     *
     * @param key metadata key
     * @param type value type
     * @param <T> value type
     * @return optional typed value
     */
    @SuppressWarnings("unchecked")
    public <T> Optional<T> getMetadata(String key, Class<T> type) {
        Object value = metadata.get(key);
        if (value == null) {
            return Optional.empty();
        }
        if (type.isInstance(value)) {
            return Optional.of((T) value);
        }
        throw new IllegalArgumentException(
            "Metadata '" + key + "' is not of type " + type.getSimpleName());
    }

    /**
     * Create default config.
     *
     * @return default config
     */
    public static OptimizationConfig defaultConfig() {
        return new OptimizationConfig(
            "unknown",
            32,  // numHeads
            32,  // numKVHeads
            128, // headDim
            1024,// seqLen
            4096,// hiddenDim
            32,  // numLayers
            Map.of()
        );
    }

    /**
     * Builder for OptimizationConfig.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder class.
     */
    public static class Builder {
        private String modelPath = "unknown";
        private int numHeads = 32;
        private int numKVHeads = 32;
        private int headDim = 128;
        private int seqLen = 1024;
        private int hiddenDim = 4096;
        private int numLayers = 32;
        private Map<String, Object> metadata = Map.of();

        public Builder modelPath(String modelPath) {
            this.modelPath = Objects.requireNonNull(modelPath);
            return this;
        }

        public Builder numHeads(int numHeads) {
            this.numHeads = numHeads;
            return this;
        }

        public Builder numKVHeads(int numKVHeads) {
            this.numKVHeads = numKVHeads;
            return this;
        }

        public Builder headDim(int headDim) {
            this.headDim = headDim;
            return this;
        }

        public Builder seqLen(int seqLen) {
            this.seqLen = seqLen;
            return this;
        }

        public Builder hiddenDim(int hiddenDim) {
            this.hiddenDim = hiddenDim;
            return this;
        }

        public Builder numLayers(int numLayers) {
            this.numLayers = numLayers;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = Map.copyOf(metadata);
            return this;
        }

        public Builder metadata(String key, Object value) {
            if (this.metadata.isEmpty()) {
                this.metadata = new java.util.HashMap<>();
            }
            ((java.util.HashMap<String, Object>)this.metadata).put(key, value);
            return this;
        }

        public OptimizationConfig build() {
            return new OptimizationConfig(
                modelPath,
                numHeads,
                numKVHeads,
                headDim,
                seqLen,
                hiddenDim,
                numLayers,
                Collections.unmodifiableMap(new java.util.HashMap<>(metadata))
            );
        }
    }
}
