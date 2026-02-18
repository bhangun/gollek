package tech.kayys.gollek.engine.inference;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.Objects;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import io.quarkus.panache.common.Page;

import tech.kayys.gollek.engine.model.Model;
import io.smallrye.mutiny.Uni;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Convert;
import jakarta.persistence.Converter;

@Entity
@Table(name = "inference_requests")
public class InferenceRequestEntity extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    public UUID id;

    @Column(unique = true, nullable = false)
    public String requestId;

    @ManyToOne
    @JoinColumn(name = "model_id", nullable = false)
    public Model model;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    public RequestStatus status;

    public String runnerName;
    public Long latencyMs;
    public Long inputSizeBytes;
    public Long outputSizeBytes;
    public String errorCode;
    public String errorMessage;

    @Convert(converter = MapToJsonConverter.class)
    @Column(columnDefinition = "text") // Using text column to store JSON string
    public Map<String, Object> metadata;

    public LocalDateTime createdAt;
    public LocalDateTime completedAt;

    public InferenceRequestEntity() {
    }

    public InferenceRequestEntity(UUID id, String requestId, Model model, RequestStatus status,
            String runnerName, Long latencyMs, Long inputSizeBytes, Long outputSizeBytes,
            String errorCode, String errorMessage, Map<String, Object> metadata,
            LocalDateTime createdAt, LocalDateTime completedAt) {
        this.id = id;
        this.requestId = requestId;
        this.model = model;
        this.status = status;
        this.runnerName = runnerName;
        this.latencyMs = latencyMs;
        this.inputSizeBytes = inputSizeBytes;
        this.outputSizeBytes = outputSizeBytes;
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.metadata = metadata;
        this.createdAt = createdAt;
        this.completedAt = completedAt;
    }

    // Getters
    public UUID getId() {
        return id;
    }

    public String getRequestId() {
        return requestId;
    }

    public Model getModel() {
        return model;
    }

    public RequestStatus getStatus() {
        return status;
    }

    public String getRunnerName() {
        return runnerName;
    }

    public Long getLatencyMs() {
        return latencyMs;
    }

    public Long getInputSizeBytes() {
        return inputSizeBytes;
    }

    public Long getOutputSizeBytes() {
        return outputSizeBytes;
    }

    public String getErrorCode() {
        return errorCode;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public LocalDateTime getCompletedAt() {
        return completedAt;
    }

    // Setters
    public void setId(UUID id) {
        this.id = id;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public void setModel(Model model) {
        this.model = model;
    }

    public void setStatus(RequestStatus status) {
        this.status = status;
    }

    public void setRunnerName(String runnerName) {
        this.runnerName = runnerName;
    }

    public void setLatencyMs(Long latencyMs) {
        this.latencyMs = latencyMs;
    }

    public void setInputSizeBytes(Long inputSizeBytes) {
        this.inputSizeBytes = inputSizeBytes;
    }

    public void setOutputSizeBytes(Long outputSizeBytes) {
        this.outputSizeBytes = outputSizeBytes;
    }

    public void setErrorCode(String errorCode) {
        this.errorCode = errorCode;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public void setCompletedAt(LocalDateTime completedAt) {
        this.completedAt = completedAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        InferenceRequestEntity that = (InferenceRequestEntity) o;
        return Objects.equals(id, that.id) &&
                Objects.equals(requestId, that.requestId) &&
                Objects.equals(model, that.model) &&
                status == that.status &&
                Objects.equals(runnerName, that.runnerName) &&
                Objects.equals(latencyMs, that.latencyMs) &&
                Objects.equals(inputSizeBytes, that.inputSizeBytes) &&
                Objects.equals(outputSizeBytes, that.outputSizeBytes) &&
                Objects.equals(errorCode, that.errorCode) &&
                Objects.equals(errorMessage, that.errorMessage) &&
                Objects.equals(metadata, that.metadata) &&
                Objects.equals(createdAt, that.createdAt) &&
                Objects.equals(completedAt, that.completedAt);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, requestId, model, status, runnerName, latencyMs,
                inputSizeBytes, outputSizeBytes, errorCode, errorMessage,
                metadata, createdAt, completedAt);
    }

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }

    public enum RequestStatus {
        PENDING, PROCESSING, COMPLETED, FAILED, TIMEOUT
    }

    // Panache queries
    public static Uni<List<InferenceRequestEntity>> findByTenant(String requestId, int page, int size) {
        return find("tenant.requestId = ?1 order by createdAt desc", requestId)
                .page(Page.of(page, size))
                .list();
    }

    /**
     * Attribute converter to convert Map to JSON string and vice versa.
     */
    @Converter
    public static class MapToJsonConverter implements AttributeConverter<Map<String, Object>, String> {

        private static final ObjectMapper objectMapper = new ObjectMapper();

        @Override
        public String convertToDatabaseColumn(Map<String, Object> attribute) {
            if (attribute == null || attribute.isEmpty()) {
                return null;
            }
            try {
                return objectMapper.writeValueAsString(attribute);
            } catch (JsonProcessingException e) {
                throw new RuntimeException("Failed to convert map to JSON", e);
            }
        }

        @Override
        public Map<String, Object> convertToEntityAttribute(String dbData) {
            if (dbData == null || dbData.trim().isEmpty()) {
                return null;
            }
            try {
                return objectMapper.readValue(dbData, new TypeReference<Map<String, Object>>() {
                });
            } catch (JsonProcessingException e) {
                throw new RuntimeException("Failed to convert JSON to map", e);
            }
        }
    }

    public static Uni<Long> countByTenantAndTimeRange(
            String requestId,
            LocalDateTime start,
            LocalDateTime end) {
        return count("tenant.requestId = ?1 and createdAt between ?2 and ?3",
                requestId, start, end);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private UUID id;
        private String requestId;
        private Model model;
        private RequestStatus status;
        private String runnerName;
        private Long latencyMs;
        private Long inputSizeBytes;
        private Long outputSizeBytes;
        private String errorCode;
        private String errorMessage;
        private Map<String, Object> metadata;
        private LocalDateTime createdAt;
        private LocalDateTime completedAt;

        public Builder id(UUID id) {
            this.id = id;
            return this;
        }

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder model(Model model) {
            this.model = model;
            return this;
        }

        public Builder status(RequestStatus status) {
            this.status = status;
            return this;
        }

        public Builder runnerName(String runnerName) {
            this.runnerName = runnerName;
            return this;
        }

        public Builder latencyMs(Long latencyMs) {
            this.latencyMs = latencyMs;
            return this;
        }

        public Builder inputSizeBytes(Long inputSizeBytes) {
            this.inputSizeBytes = inputSizeBytes;
            return this;
        }

        public Builder outputSizeBytes(Long outputSizeBytes) {
            this.outputSizeBytes = outputSizeBytes;
            return this;
        }

        public Builder errorCode(String errorCode) {
            this.errorCode = errorCode;
            return this;
        }

        public Builder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder createdAt(LocalDateTime createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder completedAt(LocalDateTime completedAt) {
            this.completedAt = completedAt;
            return this;
        }

        public InferenceRequestEntity build() {
            return new InferenceRequestEntity(id, requestId, model, status, runnerName,
                    latencyMs, inputSizeBytes, outputSizeBytes, errorCode,
                    errorMessage, metadata, createdAt, completedAt);
        }
    }
}