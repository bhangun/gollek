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

package tech.kayys.gollek.plugin.kernel;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Execution context for kernel operations with configuration, parameters,
 * and execution metadata.
 *
 * <p>Provides access to:
 * <ul>
 *   <li>Kernel configuration</li>
 *   <li>Operation parameters</li>
 *   <li>Execution metadata (timing, tracing)</li>
 *   <li>Resource management</li>
 * </ul>
 *
 * @since 2.0.0
 */
public final class KernelContext {

    private final KernelConfig config;
    private final Map<String, Object> parameters;
    private final Map<String, Object> metadata;
    private final KernelExecutionContext executionContext;

    private KernelContext(Builder builder) {
        this.config = builder.config;
        this.parameters = Collections.unmodifiableMap(builder.parameters);
        this.metadata = Collections.unmodifiableMap(builder.metadata);
        this.executionContext = builder.executionContext;
    }

    /**
     * Get kernel configuration.
     *
     * @return kernel configuration
     */
    public KernelConfig getConfig() {
        return config;
    }

    /**
     * Get operation parameters.
     *
     * @return immutable parameter map
     */
    public Map<String, Object> getParameters() {
        return parameters;
    }

    /**
     * Get parameter by name.
     *
     * @param name parameter name
     * @return optional parameter value
     */
    public Optional<Object> getParameter(String name) {
        return Optional.ofNullable(parameters.get(name));
    }

    /**
     * Get parameter by name with type conversion.
     *
     * @param name parameter name
     * @param type expected type
     * @param <T> parameter type
     * @return optional typed parameter value
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
     *
     * @return immutable metadata map
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Get metadata value by key.
     *
     * @param key metadata key
     * @return optional metadata value
     */
    public Optional<Object> getMetadataValue(String key) {
        return Optional.ofNullable(metadata.get(key));
    }

    /**
     * Get execution context.
     *
     * @return execution context
     */
    public KernelExecutionContext getExecutionContext() {
        return executionContext;
    }

    /**
     * Create a new context with additional parameter.
     *
     * @param key parameter key
     * @param value parameter value
     * @return new context with added parameter
     */
    public KernelContext withParameter(String key, Object value) {
        Map<String, Object> newParams = new java.util.HashMap<>(parameters);
        newParams.put(key, value);
        return toBuilder().parameters(newParams).build();
    }

    /**
     * Create a new context with additional metadata.
     *
     * @param key metadata key
     * @param value metadata value
     * @return new context with added metadata
     */
    public KernelContext withMetadata(String key, Object value) {
        Map<String, Object> newMetadata = new java.util.HashMap<>(metadata);
        newMetadata.put(key, value);
        return toBuilder().metadata(newMetadata).build();
    }

    /**
     * Create builder from this context.
     *
     * @return builder pre-populated with current values
     */
    public Builder toBuilder() {
        return new Builder()
            .config(config)
            .parameters(new HashMap<>(parameters))
            .metadata(new HashMap<>(metadata))
            .executionContext(executionContext);
    }

    /**
     * Create empty context with default configuration.
     *
     * @return empty context
     */
    public static KernelContext empty() {
        return builder().build();
    }

    /**
     * Create context with parameters.
     *
     * @param parameters operation parameters
     * @return context with parameters
     */
    public static KernelContext withParameters(Map<String, Object> parameters) {
        return builder().parameters(parameters).build();
    }

    /**
     * Create builder.
     *
     * @return new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for KernelContext.
     */
    public static class Builder {
        private KernelConfig config = KernelConfig.defaultConfig();
        private Map<String, Object> parameters = new HashMap<>();
        private Map<String, Object> metadata = new HashMap<>();
        private KernelExecutionContext executionContext = KernelExecutionContext.empty();

        public Builder config(KernelConfig config) {
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

        public Builder executionContext(KernelExecutionContext executionContext) {
            this.executionContext = executionContext;
            return this;
        }

        public KernelContext build() {
            return new KernelContext(this);
        }
    }
}
