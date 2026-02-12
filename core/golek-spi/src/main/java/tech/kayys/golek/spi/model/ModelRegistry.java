package tech.kayys.golek.spi.model;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.spi.auth.ApiKeyConstants;
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
         * API-key-first alias for {@link #getManifest(String, String, String)}.
         */
        default Uni<ModelManifest> getManifestByApiKey(String apiKey, String modelId, String version) {
                return getManifest(apiKey, modelId, version);
        }

        /**
         * Find models by tenant.
         */
        Uni<List<ModelManifest>> findByTenant(TenantId tenantId, Pageable pageable);

        /**
         * API-key-first alias for {@link #findByTenant(TenantId, Pageable)}.
         */
        default Uni<List<ModelManifest>> findByApiKey(String apiKey, Pageable pageable) {
                return findByTenant(new TenantId(apiKey), pageable);
        }

        /**
         * Delete a model and all its versions.
         */
        Uni<Void> deleteModel(String tenantId, String modelId);

        /**
         * API-key-first alias for {@link #deleteModel(String, String)}.
         */
        default Uni<Void> deleteModelByApiKey(String apiKey, String modelId) {
                return deleteModel(apiKey, modelId);
        }

        /**
         * Get model statistics.
         */
        Uni<ModelStats> getModelStats(String tenantId, String modelId);

        /**
         * API-key-first alias for {@link #getModelStats(String, String)}.
         */
        default Uni<ModelStats> getModelStatsByApiKey(String apiKey, String modelId) {
                return getModelStats(apiKey, modelId);
        }

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
                public String apiKey() {
                        if (tenantId == null || tenantId.isBlank()) {
                                return ApiKeyConstants.COMMUNITY_API_KEY;
                        }
                        return tenantId;
                }
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
