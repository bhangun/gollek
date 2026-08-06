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
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Execution context for runner operations with configuration and metadata.
 *
 * @since 2.0.0
 */
public final class RunnerContext {

    private final RunnerConfig config;
    private final Map<String, Object> parameters;
    private final Map<String, Object> metadata;
    private final RunnerExecutionContext executionContext;

    private RunnerContext(Builder builder) {
        this.config = builder.config;
        this.parameters = Collections.unmodifiableMap(builder.parameters);
        this.metadata = Collections.unmodifiableMap(builder.metadata);
        this.executionContext = builder.executionContext;
    }

    /**
     * Get runner configuration.
     */
    public RunnerConfig getConfig() {
        return config;
    }

    /**
     * Get operation parameters.
     */
    public Map<String, Object> getParameters() {
        return parameters;
    }

    /**
     * Get parameter by name.
     */
    public Optional<Object> getParameter(String name) {
        return Optional.ofNullable(parameters.get(name));
    }

    /**
     * Get typed parameter.
     */
    @SuppressWarnings("unchecked")
    public <T> Optional<T> getParameter(String name, Class<T> type) {
        Object value = parameters.get(name);
        if (value == null) {
            return Optional.empty();
        }
        if (type.isInstance(value)) {
            return Optional.of((T) value);
        }
        throw new IllegalArgumentException(
            "Parameter '" + name + "' is not of type " + type.getSimpleName());
    }

    /**
     * Get execution metadata.
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Get metadata value.
     */
    public Optional<Object> getMetadataValue(String key) {
        return Optional.ofNullable(metadata.get(key));
    }

    /**
     * Get execution context.
     */
    public RunnerExecutionContext getExecutionContext() {
        return executionContext;
    }

    /**
     * Create empty context.
     */
    public static RunnerContext empty() {
        return builder().build();
    }

    /**
     * Create context with parameters.
     */
    public static RunnerContext withParameters(Map<String, Object> parameters) {
        return builder().parameters(parameters).build();
    }

    /**
     * Create context from a configuration map.
     */
    public static RunnerContext fromMap(Map<String, Object> config) {
        return builder().parameters(config).build();
    }

    /**
     * Create builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for RunnerContext.
     */
    public static class Builder {
        private RunnerConfig config = RunnerConfig.defaultConfig();
        private Map<String, Object> parameters = new HashMap<>();
        private Map<String, Object> metadata = new HashMap<>();
        private RunnerExecutionContext executionContext = RunnerExecutionContext.empty();

        public Builder config(RunnerConfig config) {
            this.config = Objects.requireNonNull(config);
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters = new HashMap<>(parameters);
            return this;
        }

        public Builder parameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = new HashMap<>(metadata);
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder executionContext(RunnerExecutionContext executionContext) {
            this.executionContext = executionContext;
            return this;
        }

        public RunnerContext build() {
            return new RunnerContext(this);
        }
    }
}
