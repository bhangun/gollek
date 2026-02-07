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
import tech.kayys.golek.engine.tenant.Tenant;

@Entity
@Table(name = "models")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Model extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    public UUID id;

    @ManyToOne
    @JoinColumn(name = "tenant_id", nullable = false)
    public Tenant tenant;

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

    // Panache queries
    public static Uni<Model> findByTenantAndModelId(String tenantId, String modelId) {
        return find("tenant.tenantId = ?1 and modelId = ?2", tenantId, modelId).firstResult();
    }

    public static Uni<List<Model>> findByTenant(String tenantId) {
        return list("tenant.tenantId", tenantId);
    }

    public static Uni<List<Model>> findByStage(ModelStage stage) {
        return list("stage", stage);
    }
}
