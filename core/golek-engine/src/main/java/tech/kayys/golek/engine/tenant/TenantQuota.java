package tech.kayys.golek.engine.tenant;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

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
@Table(name = "tenant_quotas")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TenantQuota extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    public UUID id;

    @ManyToOne
    @JoinColumn(name = "tenant_id", nullable = false)
    public Tenant tenant;

    @Column(nullable = false)
    public String resourceType;

    @Column(nullable = false)
    public Long quotaLimit;

    @Column(nullable = false)
    public Long quotaUsed;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    public ResetPeriod resetPeriod;

    public LocalDateTime lastReset;
    public LocalDateTime createdAt;
    public LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        lastReset = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    public enum ResetPeriod {
        HOURLY, DAILY, MONTHLY, NONE
    }

    // Business methods
    public boolean hasQuotaAvailable(long amount) {
        return quotaUsed + amount <= quotaLimit;
    }

    public void incrementUsage(long amount) {
        this.quotaUsed += amount;
    }

    public double getUsagePercentage() {
        if (quotaLimit == 0)
            return 0.0;
        return (quotaUsed * 100.0) / quotaLimit;
    }

    // Panache queries
    public static List<TenantQuota> findByTenant(UUID tenantId) {
        return list("tenant.id", tenantId);
    }

    public static TenantQuota findByTenantAndResource(UUID tenantId, String resourceType) {
        return find("tenant.id = ?1 and resourceType = ?2", tenantId, resourceType).firstResult();
    }
}