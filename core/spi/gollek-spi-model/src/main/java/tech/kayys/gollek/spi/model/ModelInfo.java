package tech.kayys.gollek.spi.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * Model information DTO representing metadata about an available model.
 */
public final class ModelInfo {

    private final String modelId;
    private final String shortId;
    private final String name;
    private final String version;
    private final String architecture;
    private final String parameterCount;
    private final String description;
    private final Long contextLength;
    private final Integer embeddingSize;
    private final Integer inputTokenLimit;
    private final Integer outputTokenLimit;
    @Deprecated
    private final String requestId;
    private final String format;
    private final Long sizeBytes;
    private final String quantization;
    private final Instant createdAt;
    private final Instant updatedAt;
    private final Map<String, Object> metadata;

    @JsonCreator
    public ModelInfo(
            @JsonProperty("modelId") String modelId,
            @JsonProperty("shortId") String shortId,
            @JsonProperty("name") String name,
            @JsonProperty("version") String version,
            @JsonProperty("architecture") String architecture,
            @JsonProperty("parameterCount") String parameterCount,
            @JsonProperty("description") String description,
            @JsonProperty("contextLength") Long contextLength,
            @JsonProperty("embeddingSize") Integer embeddingSize,
            @JsonProperty("inputTokenLimit") Integer inputTokenLimit,
            @JsonProperty("outputTokenLimit") Integer outputTokenLimit,
            @JsonProperty("requestId") String requestId,
            @JsonProperty("format") String format,
            @JsonProperty("sizeBytes") Long sizeBytes,
            @JsonProperty("quantization") String quantization,
            @JsonProperty("createdAt") Instant createdAt,
            @JsonProperty("updatedAt") Instant updatedAt,
            @JsonProperty("metadata") Map<String, Object> metadata) {
        this.modelId = Objects.requireNonNull(modelId, "modelId is required");
        this.shortId = shortId;
        this.name = name;
        this.version = version;
        this.architecture = architecture;
        this.parameterCount = parameterCount;
        this.description = description;
        this.contextLength = contextLength;
        this.embeddingSize = embeddingSize;
        this.inputTokenLimit = inputTokenLimit;
        this.outputTokenLimit = outputTokenLimit;
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

    public String getShortId() {
        return shortId;
    }

    public String getName() {
        return name;
    }

    public String getArchitecture() {
        return architecture;
    }

    public String getParameterCount() {
        return parameterCount;
    }

    public String getDescription() {
        return description;
    }

    public Long getContextLength() {
        return contextLength;
    }

    public Integer getEmbeddingSize() {
        return embeddingSize;
    }

    public Integer getInputTokenLimit() {
        return inputTokenLimit;
    }

    public Integer getOutputTokenLimit() {
        return outputTokenLimit;
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

    public Builder toBuilder() {
        return new Builder()
                .modelId(modelId)
                .shortId(shortId)
                .name(name)
                .version(version)
                .architecture(architecture)
                .parameterCount(parameterCount)
                .description(description)
                .contextLength(contextLength)
                .embeddingSize(embeddingSize)
                .inputTokenLimit(inputTokenLimit)
                .outputTokenLimit(outputTokenLimit)
                .requestId(requestId)
                .format(format)
                .sizeBytes(sizeBytes)
                .quantization(quantization)
                .createdAt(createdAt)
                .updatedAt(updatedAt)
                .metadata(metadata);
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
        private String shortId;
        private String name;
        private String version;
        private String architecture;
        private String parameterCount;
        private String description;
        private Long contextLength;
        private Integer embeddingSize;
        private Integer inputTokenLimit;
        private Integer outputTokenLimit;
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

        public Builder shortId(String shortId) {
            this.shortId = shortId;
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

        public Builder architecture(String architecture) {
            this.architecture = architecture;
            return this;
        }

        public Builder parameterCount(String parameterCount) {
            this.parameterCount = parameterCount;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder contextLength(Long contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        public Builder embeddingSize(Integer embeddingSize) {
            this.embeddingSize = embeddingSize;
            return this;
        }

        public Builder inputTokenLimit(Integer inputTokenLimit) {
            this.inputTokenLimit = inputTokenLimit;
            return this;
        }

        public Builder outputTokenLimit(Integer outputTokenLimit) {
            this.outputTokenLimit = outputTokenLimit;
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
            return new ModelInfo(modelId, shortId, name, version, architecture, parameterCount, 
                    description, contextLength, embeddingSize, inputTokenLimit, outputTokenLimit,
                    requestId, format, sizeBytes, quantization, createdAt, updatedAt, metadata);
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
