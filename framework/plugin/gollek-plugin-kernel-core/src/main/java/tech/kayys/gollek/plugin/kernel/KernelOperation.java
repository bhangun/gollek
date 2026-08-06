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

/**
 * Kernel operation representation with name, parameters, and metadata.
 *
 * @param name Operation name (e.g., "gemm", "attention", "layer_norm")
 * @param parameters Operation parameters
 * @param metadata Additional operation metadata
 * @since 2.0.0
 */
public record KernelOperation(
    String name,
    Map<String, Object> parameters,
    Map<String, Object> metadata
) {
    /**
     * Create operation with name only.
     */
    public KernelOperation(String name) {
        this(name, Collections.emptyMap(), Collections.emptyMap());
    }

    /**
     * Create operation with name and parameters.
     */
    public KernelOperation(String name, Map<String, Object> parameters) {
        this(name, parameters, Collections.emptyMap());
    }

    /**
     * Get operation name (alias for name()).
     *
     * @return operation name
     */
    public String getName() {
        return name;
    }

    /**
     * Builder for KernelOperation.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder class for KernelOperation.
     */
    public static class Builder {
        private String name;
        private Map<String, Object> parameters = new HashMap<>();
        private Map<String, Object> metadata = new HashMap<>();

        public Builder name(String name) {
            this.name = name;
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

        public KernelOperation build() {
            Objects.requireNonNull(name, "Operation name is required");
            return new KernelOperation(
                name,
                Collections.unmodifiableMap(new HashMap<>(parameters)),
                Collections.unmodifiableMap(new HashMap<>(metadata))
            );
        }
    }
}
