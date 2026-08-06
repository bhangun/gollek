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

import java.util.Map;
import java.util.UUID;

/**
 * Handle to a loaded model instance.
 *
 * @param modelId   Unique model identifier
 * @param modelPath Path to model file
 * @param format    Model format (gguf, onnx, etc.)
 * @param loadedAt  Load timestamp
 * @param metadata  Model metadata
 * @since 2.0.0
 */
public record ModelHandle(
        String modelId,
        String modelPath,
        String format,
        long loadedAt,
        Map<String, Object> metadata) {
    /**
     * Create model handle with auto-generated ID.
     */
    public ModelHandle(String modelPath, String format, Map<String, Object> metadata) {
        this(UUID.randomUUID().toString(), modelPath, format, System.currentTimeMillis(), metadata);
    }

    /**
     * Create model handle.
     */
    public ModelHandle(String modelPath, String format) {
        this(modelPath, format, Map.of());
    }

    /**
     * Get model ID (alias for modelId()).
     */
    public String getModelId() {
        return modelId;
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
        return format;
    }

    /**
     * Get load timestamp (alias for loadedAt()).
     */
    public long getLoadedAt() {
        return loadedAt;
    }

    /**
     * Get metadata (alias for metadata()).
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Get metadata value.
     */
    public Object getMetadataValue(String key) {
        return metadata.get(key);
    }

    /**
     * Create model handle.
     */
    public static ModelHandle of(String modelPath, String format) {
        return new ModelHandle(modelPath, format);
    }

    /**
     * Create model handle with metadata.
     */
    public static ModelHandle of(String modelPath, String format, Map<String, Object> metadata) {
        return new ModelHandle(modelPath, format, metadata);
    }
}
