package tech.kayys.golek.sdk.core.model;

import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * Model information DTO representing metadata about an available model.
 */
public final class ModelInfo {

    private final String modelId;
    private final String name;
    private final String version;
    @Deprecated
    private final String requestId;
    private final String format;
    private final Long sizeBytes;
    private final String quantization;
    private final Instant createdAt;
    private final Instant updatedAt;
    private final Map<String, Object> metadata;

    public ModelInfo(
            String modelId,
            String name,
            String version,
            String requestId,
            String format,
            Long sizeBytes,
            String quantization,
            Instant createdAt,
            Instant updatedAt,
            Map<String, Object> metadata) {
        this.modelId = Objects.requireNonNull(modelId, "modelId is required");
        this.name = name;
        this.version = version;
        this.requestId = requestId;
        this.format = format;
        this.sizeBytes = sizeBytes;
        this.quantization = quantization;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
        this.metadata = metadata != null ? Map.copyOf(metadata) : Map.of();
    }

    public String getModelId() {
        return modelId;
    }

    public String getName() {
        return name;
    }

    public String getVersion() {
        return version;
    }

    public String getApiKey() {
        if (requestId == null || requestId.isBlank()) {
            return "community";
        }
        return requestId;
    }

    /**
     * @deprecated Tenant ID is resolved server-side from the API key.
     *             Client code should not rely on this field.
     */
    @Deprecated
    public String getRequestId() {
        return requestId;
    }

    public String getFormat() {
        return format;
    }

    public Long getSizeBytes() {
        return sizeBytes;
    }

    public String getQuantization() {
        return quantization;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Returns a human-readable size string.
     */
    public String getSizeFormatted() {
        if (sizeBytes == null || sizeBytes == 0) {
            return "Unknown";
        }
        if (sizeBytes < 1024) {
            return sizeBytes + " B";
        } else if (sizeBytes < 1024 * 1024) {
            return String.format("%.1f KB", sizeBytes / 1024.0);
        } else if (sizeBytes < 1024 * 1024 * 1024) {
            return String.format("%.1f MB", sizeBytes / (1024.0 * 1024));
        } else {
            return String.format("%.2f GB", sizeBytes / (1024.0 * 1024 * 1024));
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String modelId;
        private String name;
        private String version;
        @Deprecated
        private String requestId;
        private String format;
        private Long sizeBytes;
        private String quantization;
        private Instant createdAt;
        private Instant updatedAt;
        private Map<String, Object> metadata;

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        /**
         * @deprecated Tenant ID is resolved server-side from the API key.
         *             Client code should not set or rely on this value.
         */
        @Deprecated
        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.requestId = apiKey;
            return this;
        }

        public Builder format(String format) {
            this.format = format;
            return this;
        }

        public Builder sizeBytes(Long sizeBytes) {
            this.sizeBytes = sizeBytes;
            return this;
        }

        public Builder quantization(String quantization) {
            this.quantization = quantization;
            return this;
        }

        public Builder createdAt(Instant createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder updatedAt(Instant updatedAt) {
            this.updatedAt = updatedAt;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public ModelInfo build() {
            return new ModelInfo(modelId, name, version, requestId, format,
                    sizeBytes, quantization, createdAt, updatedAt, metadata);
        }
    }

    @Override
    public String toString() {
        return "ModelInfo{" +
                "modelId='" + modelId + '\'' +
                ", name='" + name + '\'' +
                ", version='" + version + '\'' +
                ", format='" + format + '\'' +
                ", size=" + getSizeFormatted() +
                '}';
    }
}
