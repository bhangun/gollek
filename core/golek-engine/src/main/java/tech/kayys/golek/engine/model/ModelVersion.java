package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

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
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "model_versions")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
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
}
