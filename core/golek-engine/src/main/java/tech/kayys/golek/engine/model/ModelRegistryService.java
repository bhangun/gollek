package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.engine.tenant.AuthenticationException;
import tech.kayys.wayang.tenant.TenantId;
import tech.kayys.golek.spi.exception.ModelException;
import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.engine.tenant.Tenant;
import tech.kayys.golek.engine.inference.InferenceRequestEntity;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Service for managing the model registry.
 */
@Slf4j
@ApplicationScoped
public class ModelRegistryService implements tech.kayys.golek.spi.model.ModelRegistry {

        @Inject
        ModelRepository modelRepository;

        @Inject
        tech.kayys.golek.spi.storage.ModelStorageService storageService;

        /**
         * Register a new model.
         */
        @Override
        @Transactional
        public Uni<ModelVersion> registerModel(ModelUploadRequest request) {
                log.info("Registering model: tenantId={}, modelId={}, version={}",
                                request.tenantId(), request.modelId(), request.version());

                return Tenant.findByTenantId(request.tenantId())
                                .onItem().ifNull().failWith(() -> new AuthenticationException(
                                                ErrorCode.AUTH_TENANT_NOT_FOUND,
                                                "Tenant not found: " + request.tenantId()))
                                .chain(tenant -> Model.findByTenantAndModelId(request.tenantId(), request.modelId())
                                                .chain(model -> {
                                                        if (model == null) {
                                                                Model newModel = Model.builder()
                                                                                .tenant(tenant)
                                                                                .modelId(request.modelId())
                                                                                .name(request.name() != null
                                                                                                ? request.name()
                                                                                                : request.modelId())
                                                                                .description(request.description())
                                                                                .framework(request.framework())
                                                                                .stage(Model.ModelStage.DEVELOPMENT)
                                                                                .tags(request.tags())
                                                                                .metadata(request.metadata())
                                                                                .createdBy(request.createdBy())
                                                                                .build();
                                                                return newModel.persist().replaceWith(newModel);
                                                        }
                                                        return Uni.createFrom().item(model);
                                                }))
                                .chain(model -> ModelVersion.findByModelAndVersion(model.id, request.version())
                                                .chain(existingVersion -> {
                                                        if (existingVersion != null) {
                                                                throw new ModelException(
                                                                                ErrorCode.MODEL_INVALID_FORMAT,
                                                                                "Model version already exists: "
                                                                                                + request.version(),
                                                                                request.modelId());
                                                        }
                                                        return storageService.uploadModel(
                                                                        request.tenantId(),
                                                                        request.modelId(),
                                                                        request.version(),
                                                                        request.modelData()).map(storageUri -> {
                                                                                String checksum = calculateChecksum(
                                                                                                request.modelData());
                                                                                ModelVersion version = ModelVersion
                                                                                                .builder()
                                                                                                .model(model)
                                                                                                .version(request.version())
                                                                                                .storageUri(storageUri)
                                                                                                .format(request.framework())
                                                                                                .checksum(checksum)
                                                                                                .sizeBytes((long) request
                                                                                                                .modelData().length)
                                                                                                .manifest(buildManifest(
                                                                                                                request))
                                                                                                .status(ModelVersion.VersionStatus.ACTIVE)
                                                                                                .build();

                                                                                version.persist();
                                                                                return version;
                                                                        });
                                                }));
        }

        /**
         * Get model manifest by ID and version.
         */
        public Uni<ModelManifest> getManifest(String tenantId, String modelId, String version) {
                return Model.findByTenantAndModelId(tenantId, modelId)
                                .onItem().ifNull().failWith(() -> new ModelException(
                                                ErrorCode.MODEL_NOT_FOUND,
                                                "Model not found: " + modelId,
                                                modelId))
                                .chain(model -> {
                                        if (version == null || version.equals("latest")) {
                                                return ModelVersion.findActiveVersions(model.id)
                                                                .map(versions -> {
                                                                        if (versions.isEmpty()) {
                                                                                throw new ModelException(
                                                                                                ErrorCode.MODEL_VERSION_NOT_FOUND,
                                                                                                "No active versions found for model: "
                                                                                                                + modelId,
                                                                                                modelId);
                                                                        }
                                                                        return manifestFromEntity(model,
                                                                                        versions.get(0));
                                                                });
                                        } else {
                                                return ModelVersion.findByModelAndVersion(model.id, version)
                                                                .onItem().ifNull().failWith(() -> new ModelException(
                                                                                ErrorCode.MODEL_VERSION_NOT_FOUND,
                                                                                "Version not found: " + version,
                                                                                modelId))
                                                                .map(v -> manifestFromEntity(model, v));
                                        }
                                });
        }

        @Override
        public Uni<List<ModelManifest>> findByTenant(TenantId tenantId, tech.kayys.golek.model.core.Pageable pageable) {
                return Model.findByTenant(tenantId.value())
                                .map(models -> models.stream()
                                                .skip((long) pageable.page() * pageable.size())
                                                .limit(pageable.size())
                                                .map(m -> manifestFromEntity(m, null))
                                                .filter(Objects::nonNull)
                                                .collect(Collectors.toList()));
        }

        @Override
        public Uni<Void> deleteModel(String tenantId, String modelId) {
                return Model.findByTenantAndModelId(tenantId, modelId)
                                .onItem().ifNull().failWith(() -> new ModelException(
                                                ErrorCode.MODEL_NOT_FOUND,
                                                "Model not found: " + modelId,
                                                modelId))
                                .chain(model -> model.delete());
        }

        /**
         * Get model statistics.
         */
        public Uni<ModelStats> getModelStats(String tenantId, String modelId) {
                return Model.findByTenantAndModelId(tenantId, modelId)
                                .onItem().ifNull().failWith(() -> new ModelException(
                                                ErrorCode.MODEL_NOT_FOUND,
                                                "Model not found: " + modelId,
                                                modelId))
                                .chain(model -> {
                                        Uni<Long> totalInferencesUni = InferenceRequestEntity.count(
                                                        "model.id = ?1 and status = ?2",
                                                        model.id,
                                                        InferenceRequestEntity.RequestStatus.COMPLETED);

                                        Uni<Long> versionCountUni = ModelVersion.count("model.id = ?1", model.id);

                                        return Uni.combine().all().unis(totalInferencesUni, versionCountUni)
                                                        .asTuple()
                                                        .map(tuple -> new ModelStats(
                                                                        modelId,
                                                                        model.stage,
                                                                        tuple.getItem2(),
                                                                        tuple.getItem1(),
                                                                        model.createdAt,
                                                                        model.updatedAt));
                                });
        }

        // ===== Helper Methods =====

        private String calculateChecksum(byte[] data) {
                try {
                        java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
                        byte[] hash = digest.digest(data);
                        return Base64.getEncoder().encodeToString(hash);
                } catch (Exception e) {
                        throw new RuntimeException("Error calculating checksum", e);
                }
        }

        private ModelManifest buildManifest(ModelUploadRequest request) {
                return ModelManifest.builder()
                                .modelId(request.modelId())
                                .name(request.name() != null ? request.name() : request.modelId())
                                .version(request.version())
                                .framework(request.framework())
                                .description(request.description())
                                .tags(Set.of(request.tags() != null ? request.tags() : new String[0]))
                                .metadata(request.metadata())
                                .build();
        }

        private ModelManifest manifestFromEntity(Model model, ModelVersion version) {
                ModelManifest.Builder builder = ModelManifest.builder()
                                .modelId(model.modelId)
                                .name(model.name)
                                .framework(model.framework)
                                .description(model.description)
                                .tags(Set.of(model.tags != null ? model.tags : new String[0]))
                                .metadata(model.metadata);

                if (version != null) {
                        builder.version(version.version);
                        if (version.manifest != null) {
                                // Merge or override from version manifest if needed
                        }
                }

                return builder.build();
        }

        // ===== DTOs =====

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
                        Model.ModelStage stage,
                        long versionCount,
                        long totalInferences,
                        LocalDateTime createdAt,
                        LocalDateTime updatedAt) {
        }

        public record ConversionJob(
                        String jobId,
                        String modelId,
                        String sourceFormat,
                        String targetFormat,
                        String status) {
        }
}
