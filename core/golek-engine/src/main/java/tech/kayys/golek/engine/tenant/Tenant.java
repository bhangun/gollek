package tech.kayys.golek.engine.tenant;

import io.smallrye.mutiny.Uni;
import io.hypersistence.utils.hibernate.type.json.JsonBinaryType;
import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import io.quarkus.panache.common.Page;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.Type;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Complete entity definitions for inference platform.
 * 
 * @author bhangun
 * @since 1.0.0
 */

// ===== TENANT ENTITY =====

@Entity
@Table(name = "tenants")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Tenant extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    public UUID id;

    @Column(unique = true, nullable = false)
    public String tenantId;

    @Column(nullable = false)
    public String name;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    public TenantStatus status;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    public TenantTier tier;

    @Type(JsonBinaryType.class)
    @Column(columnDefinition = "jsonb")
    public Map<String, Object> metadata;

    @Column(updatable = false)
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

    public enum TenantStatus {
        ACTIVE, SUSPENDED, DELETED
    }

    public enum TenantTier {
        FREE, BASIC, PRO, ENTERPRISE
    }

    // Panache queries
    public static Uni<Tenant> findByTenantId(String tenantId) {
        return find("tenantId", tenantId).firstResult();
    }

    public static Uni<List<Tenant>> findActive() {
        return list("status", TenantStatus.ACTIVE);
    }
}