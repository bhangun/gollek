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
import tech.kayys.golek.spi.model.ModelRegistry;
import com.fasterxml.jackson.core.type.TypeReference;
import tech.kayys.golek.spi.storage.ModelStorageService;
import tech.kayys.golek.spi.model.Pageable;

import com.fasterxml.jackson.databind.ObjectMapper;
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
        ModelStorageService storageService;

        @Inject
        ObjectMapper objectMapper;

        /**
         * Register a new model.
         */
        @Override
        @Transactional
        public Uni<ModelManifest> registerModel(ModelRegistry.ModelUploadRequest uploadRequest) {
                log.info("Registering model: tenantId={}, modelId={}, version={}",
                                uploadRequest.tenantId(), uploadRequest.modelId(), uploadRequest.version());

                return Tenant.findByTenantId(uploadRequest.tenantId())
                                .onItem().ifNull().failWith(() -> new AuthenticationException(
                                                ErrorCode.AUTH_TENANT_NOT_FOUND,
                                                "Tenant not found: " + uploadRequest.tenantId()))
                                .chain(tenant -> Model
                                                .findByTenantAndModelId(uploadRequest.tenantId(),
                                                                uploadRequest.modelId())
                                                .chain(model -> {
                                                        if (model == null) {
                                                                Model newModel = Model.builder()
                                                                                .tenant(tenant)
                                                                                .modelId(uploadRequest.modelId())
                                                                                .name(uploadRequest.name() != null
                                                                                                ? uploadRequest.name()
                                                                                                : uploadRequest.modelId())
                                                                                .description(uploadRequest
                                                                                                .description())
                                                                                .framework(uploadRequest.framework())
                                                                                .stage(Model.ModelStage.DEVELOPMENT)
                                                                                .tags(uploadRequest.tags())
                                                                                .metadata(uploadRequest.metadata())
                                                                                .createdBy(uploadRequest.createdBy())
                                                                                .build();
                                                                return newModel.persist().replaceWith(newModel);
                                                        }
                                                        return Uni.createFrom().item(model);
                                                }))
                                .chain(model -> ModelVersion.findByModelAndVersion(model.id, uploadRequest.version())
                                                .chain(existingVersion -> {
                                                        if (existingVersion != null) {
                                                                throw new ModelException(
                                                                                ErrorCode.MODEL_INVALID_FORMAT,
                                                                                "Model version already exists: "
                                                                                                + uploadRequest.version(),
                                                                                uploadRequest.modelId());
                                                        }
                                                        return storageService.uploadModel(
                                                                        uploadRequest.tenantId(),
                                                                        uploadRequest.modelId(),
                                                                        uploadRequest.version(),
                                                                        uploadRequest.modelData()).map(storageUri -> {
                                                                                String checksum = calculateChecksum(
                                                                                                uploadRequest.modelData());
                                                                                ModelVersion version = ModelVersion
                                                                                                .builder()
                                                                                                .model(model)
                                                                                                .version(uploadRequest
                                                                                                                .version())
                                                                                                .storageUri(storageUri)
                                                                                                .format(uploadRequest
                                                                                                                .framework())
                                                                                                .checksum(checksum)
                                                                                                .sizeBytes((long) uploadRequest
                                                                                                                .modelData().length)
                                                                                                .manifest(objectMapper
                                                                                                                .convertValue(buildManifest(
                                                                                                                                uploadRequest),
                                                                                                                                new TypeReference<Map<String, Object>>() {
                                                                                                                                }))
                                                                                                .status(ModelVersion.VersionStatus.ACTIVE)
                                                                                                .build();

                                                                                version.persist();
                                                                                return toManifest(model);
                                                                        });
                                                }));
        }

        /**
         * Get model manifest by ID and version.
         */
        @Override
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
        public Uni<List<ModelManifest>> findByTenant(TenantId tenantId, Pageable pageable) {
                return Model.findByTenant(tenantId.value())
                                .map(models -> models.stream()
                                                .skip((long) pageable.page() * pageable.size())
                                                .limit(pageable.size())
                                                .map(m -> toManifest(m)) // Changed to toManifest
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
        @Override
        public Uni<ModelRegistry.ModelStats> getModelStats(String tenantId, String modelId) {
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
                                                        .map(tuple -> new ModelRegistry.ModelStats(
                                                                        modelId,
                                                                        model.stage.name(),
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

        private ModelManifest buildManifest(ModelRegistry.ModelUploadRequest request) {
                return ModelManifest.builder()
                                .modelId(request.modelId())
                                .name(request.name() != null ? request.name() : request.modelId())
                                .version(request.version())
                                .tenantId(request.tenantId())
                                .metadata(request.metadata())
                                .artifacts(Collections.emptyMap())
                                .supportedDevices(Collections.emptyList())
                                .createdAt(java.time.Instant.now())
                                .updatedAt(java.time.Instant.now())
                                .build();
        }

        private ModelManifest toManifest(Model model) {
                return manifestFromEntity(model, null);
        }

        private ModelManifest manifestFromEntity(Model model, ModelVersion version) {
                ModelManifest.ModelManifestBuilder builder = ModelManifest.builder()
                                .modelId(model.modelId)
                                .name(model.name)
                                .tenantId(model.tenant.tenantId)
                                .metadata(model.metadata)
                                .artifacts(Collections.emptyMap())
                                .supportedDevices(Collections.emptyList())
                                .createdAt(model.createdAt.toInstant(java.time.ZoneOffset.UTC))
                                .updatedAt(model.updatedAt.toInstant(java.time.ZoneOffset.UTC));

                if (version != null) {
                        builder.version(version.version);
                } else {
                        builder.version("latest");
                }

                return builder.build();
        }

        public record ConversionJob(
                        String jobId,
                        String modelId,
                        String sourceFormat,
                        String targetFormat,
                        String status) {
        }
}
