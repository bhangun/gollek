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

package tech.kayys.gollek.plugin.runner;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Request to load a model into the runner.
 *
 * @param modelPath Path to model file
 * @param format Model format (optional, auto-detected if null)
 * @param config Runner configuration
 * @param metadata Additional metadata
 * @since 2.0.0
 */
public record ModelLoadRequest(
    String modelPath,
    String format,
    RunnerConfig config,
    Map<String, Object> metadata
) {
    /**
     * Create request with model path only.
     */
    public ModelLoadRequest(String modelPath) {
        this(modelPath, null, RunnerConfig.defaultConfig(), Collections.emptyMap());
    }

    /**
     * Create request with model path and config.
     */
    public ModelLoadRequest(String modelPath, RunnerConfig config) {
        this(modelPath, null, config, Collections.emptyMap());
    }

    /**
     * Get model path (alias for modelPath()).
     */
    public String getModelPath() {
        return modelPath;
    }

    /**
     * Get format (alias for format()).
     */
    public String getFormat() {
        return format != null ? format : RunnerPlugin.detectFormat(modelPath);
    }

    /**
     * Get config (alias for config()).
     */
    public RunnerConfig getConfig() {
        return config;
    }

    /**
     * Get metadata (alias for metadata()).
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Builder for ModelLoadRequest.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder class for ModelLoadRequest.
     */
    public static class Builder {
        private String modelPath;
        private String format;
        private RunnerConfig config = RunnerConfig.defaultConfig();
        private Map<String, Object> metadata = Collections.emptyMap();

        public Builder modelPath(String modelPath) {
            this.modelPath = Objects.requireNonNull(modelPath);
            return this;
        }

        public Builder format(String format) {
            this.format = format;
            return this;
        }

        public Builder config(RunnerConfig config) {
            this.config = Objects.requireNonNull(config);
            return this;
        }

        public Builder metadata(String key, Object value) {
            if (this.metadata.isEmpty()) {
                this.metadata = new java.util.HashMap<>();
            }
            ((java.util.HashMap<String, Object>)this.metadata).put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = Map.copyOf(metadata);
            return this;
        }

        public ModelLoadRequest build() {
            Objects.requireNonNull(modelPath, "Model path is required");
            return new ModelLoadRequest(
                modelPath,
                format,
                config,
                Collections.unmodifiableMap(new java.util.HashMap<>(metadata))
            );
        }
    }
}
