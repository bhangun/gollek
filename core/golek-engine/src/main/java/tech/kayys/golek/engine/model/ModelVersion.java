package tech.kayys.golek.engine.model;

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
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;

@Entity
@Table(name = "model_versions")
public class ModelVersion extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    public UUID id;

    @ManyToOne
    @JoinColumn(name = "model_id", nullable = false)
    public Model model;

    @Column(nullable = false)
    public String version;

    @Column(nullable = false)
    public String storageUri;

    @Column(nullable = false)
    public String format;

    public String checksum;
    public Long sizeBytes;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(columnDefinition = "jsonb")
    public Map<String, Object> manifest;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    public VersionStatus status;

    public LocalDateTime createdAt;
    public LocalDateTime updatedAt;

    public ModelVersion() {}

    public ModelVersion(UUID id, Model model, String version, String storageUri, String format, 
                        String checksum, Long sizeBytes, Map<String, Object> manifest, 
                        VersionStatus status, LocalDateTime createdAt, LocalDateTime updatedAt) {
        this.id = id;
        this.model = model;
        this.version = version;
        this.storageUri = storageUri;
        this.format = format;
        this.checksum = checksum;
        this.sizeBytes = sizeBytes;
        this.manifest = manifest;
        this.status = status;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    // Getters
    public UUID getId() {
        return id;
    }

    public Model getModel() {
        return model;
    }

    public String getVersion() {
        return version;
    }

    public String getStorageUri() {
        return storageUri;
    }

    public String getFormat() {
        return format;
    }

    public String getChecksum() {
        return checksum;
    }

    public Long getSizeBytes() {
        return sizeBytes;
    }

    public Map<String, Object> getManifest() {
        return manifest;
    }

    public VersionStatus getStatus() {
        return status;
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

    public void setModel(Model model) {
        this.model = model;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public void setStorageUri(String storageUri) {
        this.storageUri = storageUri;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public void setChecksum(String checksum) {
        this.checksum = checksum;
    }

    public void setSizeBytes(Long sizeBytes) {
        this.sizeBytes = sizeBytes;
    }

    public void setManifest(Map<String, Object> manifest) {
        this.manifest = manifest;
    }

    public void setStatus(VersionStatus status) {
        this.status = status;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ModelVersion that = (ModelVersion) o;
        return Objects.equals(id, that.id) &&
               Objects.equals(model, that.model) &&
               Objects.equals(version, that.version) &&
               Objects.equals(storageUri, that.storageUri) &&
               Objects.equals(format, that.format) &&
               Objects.equals(checksum, that.checksum) &&
               Objects.equals(sizeBytes, that.sizeBytes) &&
               Objects.equals(manifest, that.manifest) &&
               status == that.status &&
               Objects.equals(createdAt, that.createdAt) &&
               Objects.equals(updatedAt, that.updatedAt);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, model, version, storageUri, format, checksum, sizeBytes, 
                           manifest, status, createdAt, updatedAt);
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

    public enum VersionStatus {
        ACTIVE, DEPRECATED, DELETED
    }

    // Panache queries
    public static Uni<ModelVersion> findByModelAndVersion(UUID modelId, String version) {
        return find("model.id = ?1 and version = ?2", modelId, version).firstResult();
    }

    public static Uni<List<ModelVersion>> findActiveVersions(UUID modelId) {
        return list("model.id = ?1 and status = ?2 order by createdAt desc",
                modelId, VersionStatus.ACTIVE);
    }
    
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private UUID id;
        private Model model;
        private String version;
        private String storageUri;
        private String format;
        private String checksum;
        private Long sizeBytes;
        private Map<String, Object> manifest;
        private VersionStatus status;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;

        public Builder id(UUID id) {
            this.id = id;
            return this;
        }

        public Builder model(Model model) {
            this.model = model;
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder storageUri(String storageUri) {
            this.storageUri = storageUri;
            return this;
        }

        public Builder format(String format) {
            this.format = format;
            return this;
        }

        public Builder checksum(String checksum) {
            this.checksum = checksum;
            return this;
        }

        public Builder sizeBytes(Long sizeBytes) {
            this.sizeBytes = sizeBytes;
            return this;
        }

        public Builder manifest(Map<String, Object> manifest) {
            this.manifest = manifest;
            return this;
        }

        public Builder status(VersionStatus status) {
            this.status = status;
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

        public ModelVersion build() {
            return new ModelVersion(id, model, version, storageUri, format, checksum, 
                                  sizeBytes, manifest, status, createdAt, updatedAt);
        }
    }
}
