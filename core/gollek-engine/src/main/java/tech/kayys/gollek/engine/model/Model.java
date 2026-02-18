package tech.kayys.gollek.engine.model;

import io.smallrye.mutiny.Uni;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.Objects;

import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;
import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;
import tech.kayys.gollek.spi.model.ModelManifest;

@Entity
@Table(name = "models")
public class Model extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    public UUID id;

    @Column(nullable = false)
    public String apiKey;

    @Column(nullable = false)
    public String requestId;

    @Column(nullable = false)
    public String modelId;

    @Column(nullable = false)
    public String name;

    public String description;

    @Column(nullable = false)
    public String framework;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    public ModelStage stage;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(columnDefinition = "jsonb")
    public String[] tags;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(columnDefinition = "jsonb")
    public Map<String, Object> metadata;

    public String createdBy;
    public LocalDateTime createdAt;
    public LocalDateTime updatedAt;

    public Model() {
    }

    public Model(UUID id, String apiKey, String requestId, String modelId, String name, String description,
            String framework,
            ModelStage stage, String[] tags, Map<String, Object> metadata, String createdBy,
            LocalDateTime createdAt, LocalDateTime updatedAt) {
        this.id = id;
        this.apiKey = apiKey;
        this.requestId = requestId;
        this.modelId = modelId;
        this.name = name;
        this.description = description;
        this.framework = framework;
        this.stage = stage;
        this.tags = tags;
        this.metadata = metadata;
        this.createdBy = createdBy;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    // Getters
    public UUID getId() {
        return id;
    }

    public String getApiKey() {
        return apiKey;
    }

    public String getRequestId() {
        return requestId;
    }

    public String getModelId() {
        return modelId;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public String getFramework() {
        return framework;
    }

    public ModelStage getStage() {
        return stage;
    }

    public String[] getTags() {
        return tags;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public String getCreatedBy() {
        return createdBy;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    // Setters
    public void setId(UUID id) {
        this.id = id;
    }

    public void setApiKey(String apiKey) {
        this.apiKey = apiKey;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public void setModelId(String modelId) {
        this.modelId = modelId;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public void setFramework(String framework) {
        this.framework = framework;
    }

    public void setStage(ModelStage stage) {
        this.stage = stage;
    }

    public void setTags(String[] tags) {
        this.tags = tags;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    public void setCreatedBy(String createdBy) {
        this.createdBy = createdBy;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        Model model = (Model) o;
        return Objects.equals(id, model.id) &&
                Objects.equals(apiKey, model.apiKey) &&
                Objects.equals(requestId, model.requestId) &&
                Objects.equals(modelId, model.modelId) &&
                Objects.equals(name, model.name) &&
                Objects.equals(description, model.description) &&
                Objects.equals(framework, model.framework) &&
                stage == model.stage &&
                java.util.Arrays.equals(tags, model.tags) &&
                Objects.equals(metadata, model.metadata) &&
                Objects.equals(createdBy, model.createdBy) &&
                Objects.equals(createdAt, model.createdAt) &&
                Objects.equals(updatedAt, model.updatedAt);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(id, apiKey, requestId, modelId, name, description, framework, stage, metadata,
                createdBy,
                createdAt, updatedAt);
        result = 31 * result + java.util.Arrays.hashCode(tags);
        return result;
    }

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    public enum ModelStage {
        DEVELOPMENT, STAGING, PRODUCTION, DEPRECATED, ARCHIVED
    }

    public ModelManifest toManifest() {
        return ModelManifest.builder()
                .modelId(this.modelId)
                .name(this.name)
                .version("latest")
                .path(this.modelId)
                .apiKey(this.apiKey)
                .requestId(this.requestId)
                .metadata(this.metadata)
                .createdAt(this.createdAt != null ? this.createdAt.atZone(java.time.ZoneId.systemDefault()).toInstant()
                        : null)
                .updatedAt(this.updatedAt != null ? this.updatedAt.atZone(java.time.ZoneId.systemDefault()).toInstant()
                        : null)
                .build();
    }

    // Panache queries
    public static Uni<Model> findByTenantAndModelId(String requestId, String modelId) {
        return find("requestId = ?1 and modelId = ?2", requestId, modelId).firstResult();
    }

    public static Uni<List<Model>> findByTenant(String requestId) {
        return list("requestId", requestId);
    }

    public static Uni<List<Model>> findByStage(ModelStage stage) {
        return list("stage", stage);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private UUID id;
        private String apiKey;
        private String requestId;
        private String modelId;
        private String name;
        private String description;
        private String framework;
        private ModelStage stage;
        private String[] tags;
        private Map<String, Object> metadata;
        private String createdBy;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;

        public Builder id(UUID id) {
            this.id = id;
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder framework(String framework) {
            this.framework = framework;
            return this;
        }

        public Builder stage(ModelStage stage) {
            this.stage = stage;
            return this;
        }

        public Builder tags(String[] tags) {
            this.tags = tags;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder createdBy(String createdBy) {
            this.createdBy = createdBy;
            return this;
        }

        public Builder createdAt(LocalDateTime createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder updatedAt(LocalDateTime updatedAt) {
            this.updatedAt = updatedAt;
            return this;
        }

        public Model build() {
            return new Model(id, apiKey, requestId, modelId, name, description, framework, stage, tags, metadata,
                    createdBy,
                    createdAt, updatedAt);
        }
    }
}
