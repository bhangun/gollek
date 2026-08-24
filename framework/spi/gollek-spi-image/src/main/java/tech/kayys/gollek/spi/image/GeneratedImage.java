package tech.kayys.gollek.spi.image;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Result representing a generated image.
 */
public record GeneratedImage(
        String requestId,
        byte[] data,
        String mimeType,
        int width,
        int height,
        String modelId,
        long generationTimeMs,
        Map<String, Object> metadata) {

    public GeneratedImage {
        requestId = Objects.requireNonNull(requestId, "requestId must not be null");
        data = Objects.requireNonNull(data, "data must not be null");
        mimeType = (mimeType == null || mimeType.isBlank()) ? "image/png" : mimeType;
        modelId = Objects.requireNonNullElse(modelId, "unknown-model");
        metadata = metadata == null ? Collections.emptyMap() : Collections.unmodifiableMap(metadata);
    }

    public static GeneratedImage ofPng(
            String requestId,
            byte[] data,
            int width,
            int height,
            String modelId,
            long generationTimeMs) {
        return new GeneratedImage(requestId, data, "image/png", width, height, modelId, generationTimeMs, Map.of());
    }

    public static GeneratedImage of(
            String requestId,
            byte[] data,
            String mimeType,
            int width,
            int height,
            String modelId,
            long generationTimeMs,
            Map<String, Object> metadata) {
        return new GeneratedImage(requestId, data, mimeType, width, height, modelId, generationTimeMs, metadata);
    }
}
