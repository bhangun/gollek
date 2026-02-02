package tech.kayys.golek.engine.inference;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonSubTypes.Type;

import io.hypersistence.utils.hibernate.type.json.JsonBinaryType;
import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import io.quarkus.panache.common.Page;
import tech.kayys.golek.engine.tenant.Tenant;
import tech.kayys.golek.engine.model.Model;
import lombok.EqualsAndHashCode;
import io.smallrye.mutiny.Uni;

@Entity
@Table(name = "inference_requests")
@Data
@Builder
@EqualsAndHashCode(callSuper = false)
@NoArgsConstructor
@AllArgsConstructor
public class InferenceRequestEntity extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    public UUID id;

    @Column(unique = true, nullable = false)
    public String requestId;

    @ManyToOne
    @JoinColumn(name = "tenant_id", nullable = false)
    public Tenant tenant;

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

    @Type(JsonBinaryType.class)
    @Column(columnDefinition = "jsonb")
    public Map<String, Object> metadata;

    public LocalDateTime createdAt;
    public LocalDateTime completedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }

    public enum RequestStatus {
        PENDING, PROCESSING, COMPLETED, FAILED, TIMEOUT
    }

    // Panache queries
    public static Uni<List<InferenceRequestEntity>> findByTenant(String tenantId, int page, int size) {
        return find("tenant.tenantId = ?1 order by createdAt desc", tenantId)
                .page(Page.of(page, size))
                .list();
    }

    public static Uni<Long> countByTenantAndTimeRange(
            String tenantId,
            LocalDateTime start,
            LocalDateTime end) {
        return count("tenant.tenantId = ?1 and createdAt between ?2 and ?3",
                tenantId, start, end);
    }
}