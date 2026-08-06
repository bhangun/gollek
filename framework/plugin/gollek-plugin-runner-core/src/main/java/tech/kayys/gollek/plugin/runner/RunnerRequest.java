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
import java.util.Optional;

/**
 * Runner request with type, parameters, and metadata.
 *
 * @param type Request type (INFER, EMBED, etc.)
 * @param inferenceRequest Inference request (for INFER type)
 * @param embeddingRequest Embedding request (for EMBED type)
 * @param parameters Request parameters
 * @param metadata Additional metadata
 * @since 2.0.0
 */
public record RunnerRequest(
    RequestType type,
    tech.kayys.gollek.spi.inference.InferenceRequest inferenceRequest,
    tech.kayys.gollek.spi.embedding.EmbeddingRequest embeddingRequest,
    Map<String, Object> parameters,
    Map<String, Object> metadata
) {
    /**
     * Create request with type only.
     */
    public RunnerRequest(RequestType type) {
        this(type, null, null, Collections.emptyMap(), Collections.emptyMap());
    }

    /**
     * Create request with type and parameters.
     */
    public RunnerRequest(RequestType type, Map<String, Object> parameters) {
        this(type, null, null, parameters, Collections.emptyMap());
    }

    /**
     * Get request type (alias for type()).
     *
     * @return request type
     */
    public RequestType getType() {
        return type;
    }

    /**
     * Get inference request.
     *
     * @return optional inference request
     */
    public Optional<tech.kayys.gollek.spi.inference.InferenceRequest> getInferenceRequest() {
        return Optional.ofNullable(inferenceRequest);
    }

    /**
     * Get embedding request.
     *
     * @return optional embedding request
     */
    public Optional<tech.kayys.gollek.spi.embedding.EmbeddingRequest> getEmbeddingRequest() {
        return Optional.ofNullable(embeddingRequest);
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
     * Get typed parameter.
     *
     * @param name parameter name
     * @param type parameter type
     * @param <T> parameter type
     * @return optional typed parameter
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
     * Builder for RunnerRequest.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder class for RunnerRequest.
     */
    public static class Builder {
        private RequestType type;
        private tech.kayys.gollek.spi.inference.InferenceRequest inferenceRequest;
        private tech.kayys.gollek.spi.embedding.EmbeddingRequest embeddingRequest;
        private Map<String, Object> parameters = new java.util.HashMap<>();
        private Map<String, Object> metadata = new java.util.HashMap<>();

        public Builder type(RequestType type) {
            this.type = Objects.requireNonNull(type);
            return this;
        }

        public Builder inferenceRequest(tech.kayys.gollek.spi.inference.InferenceRequest request) {
            this.inferenceRequest = request;
            return this;
        }

        public Builder embeddingRequest(tech.kayys.gollek.spi.embedding.EmbeddingRequest request) {
            this.embeddingRequest = request;
            return this;
        }

        public Builder parameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters.putAll(parameters);
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata.putAll(metadata);
            return this;
        }

        public RunnerRequest build() {
            Objects.requireNonNull(type, "Request type is required");
            return new RunnerRequest(
                type,
                inferenceRequest,
                embeddingRequest,
                Collections.unmodifiableMap(new java.util.HashMap<>(parameters)),
                Collections.unmodifiableMap(new java.util.HashMap<>(metadata))
            );
        }
    }
}
