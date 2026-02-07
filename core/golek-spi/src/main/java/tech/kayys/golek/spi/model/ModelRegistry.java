package tech.kayys.golek.spi.model;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.tenant.TenantId;
import java.util.List;
import java.time.LocalDateTime;
import java.util.Map;

/**
 * Registry for managing models and their versions.
 */
public interface ModelRegistry {

    /**
     * Register a new model version.
     */
    Uni<ModelManifest> registerModel(ModelUploadRequest request);

    /**
     * Get model manifest by ID and version.
     */
    Uni<ModelManifest> getManifest(String tenantId, String modelId, String version);

    /**
     * Find models by tenant.
     */
    Uni<List<ModelManifest>> findByTenant(TenantId tenantId, Pageable pageable);

    /**
     * Delete a model and all its versions.
     */
    Uni<Void> deleteModel(String tenantId, String modelId);

    /**
     * Get model statistics.
     */
    Uni<ModelStats> getModelStats(String tenantId, String modelId);

    // DTOs for SPI

    public record ModelUploadRequest(
            String tenantId,
            String modelId,
            String version,
            String name,
            String description,
            String framework,
            byte[] modelData,
            String[] tags,
            Map<String, Object> metadata,
            Map<String, Object> inputSchema,
            Map<String, Object> outputSchema,
            String createdBy) {
    }

    public record ModelStats(
            String modelId,
            String stage, // Use String instead of Model.ModelStage to avoid engine dependency in SPI
            long versionCount,
            long totalInferences,
            LocalDateTime createdAt,
            LocalDateTime updatedAt) {
    }
}
