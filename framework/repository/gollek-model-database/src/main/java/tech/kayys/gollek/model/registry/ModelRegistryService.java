package tech.kayys.gollek.registry.service;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.alkhawarizm.spi.model.ModelManifest;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.auth.ApiKeyConstants;
import tech.kayys.alkhawarizm.spi.model.ModelRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelStatsProvider;

import com.fasterxml.jackson.core.type.TypeReference;
import tech.kayys.alkhawarizm.spi.model.Pageable;
import tech.kayys.alkhawarizm.spi.model.ModelStorageService;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Service for managing the model registry.
 */
@ApplicationScoped
public class ModelRegistryService implements ModelRegistry {

    private static final Logger log = LoggerFactory.getLogger(ModelRegistryService.class);

    @Inject
    Instance<ModelStorageService> storageService;

    @Inject
    ObjectMapper objectMapper;

    @Inject
    Instance<ModelStatsProvider> modelStatsProvider;

    @Inject
    tech.kayys.gollek.registry.repository.ModelRepository modelRepository;

    @Inject
    tech.kayys.gollek.registry.repository.ModelVersionRepository modelVersionRepository;

    /**
     * Register a new model.
     */
    @Override
    @Transactional
    public Uni<ModelManifest> registerModel(ModelRegistry.ModelUploadRequest uploadRequest) {
        // Compatibility entry-point: tenant must come from API-key context in higher
        // layers.
        return registerModel(uploadRequest, ApiKeyConstants.COMMUNITY_API_KEY);
    }

    @Transactional
    public Uni<ModelManifest> registerModel(
            ModelRegistry.ModelUploadRequest uploadRequest,
            RequestContext requestContext) {
        return registerModel(uploadRequest, resolveRequestId(requestContext));
    }

    private Uni<ModelManifest> registerModel(
            ModelRegistry.ModelUploadRequest uploadRequest,
            String requestId) {
        log.info("Registering model: requestId={}, modelId={}, version={}",
                requestId, uploadRequest.modelId(), uploadRequest.version());

        return modelRepository
                .findByTenantAndModelId(requestId, uploadRequest.modelId())
                .chain(model -> {
                    if (model == null) {
                        tech.kayys.gollek.registry.model.Model newModel = tech.kayys.gollek.registry.model.Model
                                .builder()
                                .apiKey(ApiKeyConstants.COMMUNITY_API_KEY)
                                .requestId(requestId)
                                .modelId(uploadRequest.modelId())
                                .name(uploadRequest.name() != null
                                        ? uploadRequest.name()
                                        : uploadRequest.modelId())
                                .description(uploadRequest
                                        .description())
                                .framework(uploadRequest.framework())
                                .stage(tech.kayys.gollek.registry.model.Model.ModelStage.DEVELOPMENT)
                                .tags(uploadRequest.tags())
                                .metadata(uploadRequest.metadata())
                                .createdBy(uploadRequest.createdBy())
                                .build();
                        return newModel.persist().replaceWith(newModel);
                    }
                    return Uni.createFrom().item(model);
                })
                .chain(model -> modelVersionRepository.findByModelAndVersion(model.id, uploadRequest.version())
                        .chain(existingVersion -> {
                            if (existingVersion != null) {
                                throw new RuntimeException(
                                        "Model version already exists: "
                                                + uploadRequest.version());
                            }
                            return storageService().uploadModelByApiKey(
                                    requestId,
                                    uploadRequest.modelId(),
                                    uploadRequest.version(),
                                    uploadRequest.modelData()).map(storageUri -> {
                                        String checksum = calculateChecksum(
                                                uploadRequest.modelData());
                                        tech.kayys.gollek.registry.model.ModelVersion version = tech.kayys.gollek.registry.model.ModelVersion
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
                                                                uploadRequest,
                                                                requestId,
                                                                storageUri,
                                                                checksum,
                                                                (long) uploadRequest
                                                                        .modelData().length),
                                                                new TypeReference<Map<String, Object>>() {
                                                                }))
                                                .status(tech.kayys.gollek.registry.model.ModelVersion.VersionStatus.ACTIVE)
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
    public Uni<ModelManifest> getManifest(String requestId, String modelId, String version) {
        return modelRepository.findByTenantAndModelId(requestId, modelId)
                .onItem().ifNull().failWith(() -> new RuntimeException(
                        "Model not found: " + modelId))
                .chain(model -> {
                    if (version == null || version.equals("latest")) {
                        return modelVersionRepository.findActiveVersions(model.id)
                                .map(versions -> {
                                    if (versions.isEmpty()) {
                                        throw new RuntimeException(
                                                "No active versions found for model: "
                                                        + modelId);
                                    }
                                    return manifestFromEntity(model,
                                            versions.get(0));
                                });
                    } else {
                        return modelVersionRepository.findByModelAndVersion(model.id, version)
                                .onItem().ifNull().failWith(() -> new RuntimeException(
                                        "Version not found: " + version + " for model: " + modelId))
                                .map(v -> manifestFromEntity(model, v));
                    }
                });
    }

    @Override
    public Uni<List<ModelManifest>> findByTenant(String requestId, Pageable pageable) {
        return modelRepository.findByTenant(resolveRequestId(requestId))
                .map(models -> models.stream()
                        .skip((long) pageable.page() * pageable.size())
                        .limit(pageable.size())
                        .map(m -> toManifest(m))
                        .filter(Objects::nonNull)
                        .collect(Collectors.toList()));
    }

    @Override
    public Uni<Void> deleteModel(String requestId, String modelId) {
        return modelRepository.findByTenantAndModelId(requestId, modelId)
                .onItem().ifNull().failWith(() -> new RuntimeException(
                        "Model not found: " + modelId))
                .chain(model -> model.delete());
    }

    /**
     * Get model statistics.
     */
    @Override
    public Uni<ModelRegistry.ModelStats> getModelStats(String requestId, String modelId) {
        return modelRepository.findByTenantAndModelId(requestId, modelId)
                .onItem().ifNull().failWith(() -> new RuntimeException(
                        "Model not found: " + modelId))
                .chain(model -> {
                    if (modelStatsProvider.isResolvable()) {
                        return modelStatsProvider.get().getModelStats(requestId, modelId);
                    } else {
                        // Fallback: return basic stats without inference count
                        return modelVersionRepository.findActiveVersions(model.id)
                                .map(versions -> new ModelRegistry.ModelStats(
                                        modelId,
                                        model.stage.name(),
                                        versions.size(),
                                        0L,
                                        model.createdAt,
                                        model.updatedAt));
                    }
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

    private ModelStorageService storageService() {
        if (storageService.isResolvable()) {
            return storageService.get();
        }
        throw new IllegalStateException(
                "No ModelStorageService bean available. Enable gollek-engine or a storage plugin.");
    }

    private ModelManifest buildManifest(
            ModelRegistry.ModelUploadRequest request,
            String requestId,
            String storageUri,
            String checksum,
            Long sizeBytes) {
        Map<tech.kayys.alkhawarizm.core.model.ModelFormat, tech.kayys.alkhawarizm.spi.model.ArtifactLocation> artifacts = buildArtifacts(
                request.framework(),
                storageUri,
                checksum,
                sizeBytes);

        return ModelManifest.builder()
                .modelId(request.modelId())
                .name(request.name() != null ? request.name() : request.modelId())
                .version(request.version())
                .path(storageUri)
                .apiKey(ApiKeyConstants.COMMUNITY_API_KEY)
                .requestId(requestId)
                .metadata(request.metadata())
                .artifacts(artifacts)
                .supportedDevices(Collections.emptyList())
                .createdAt(java.time.Instant.now())
                .updatedAt(java.time.Instant.now())
                .build();
    }

    private ModelManifest toManifest(tech.kayys.gollek.registry.model.Model model) {
        return manifestFromEntity(model, null);
    }

    private ModelManifest manifestFromEntity(tech.kayys.gollek.registry.model.Model model,
            tech.kayys.gollek.registry.model.ModelVersion version) {
        ModelManifest.Builder builder = ModelManifest.builder()
                .modelId(model.modelId)
                .name(model.name)
                .path(version != null ? version.storageUri : model.modelId)
                .apiKey(model.apiKey)
                .requestId(model.requestId)
                .metadata(model.metadata)
                .artifacts(version != null
                        ? buildArtifacts(version.format, version.storageUri, version.checksum,
                                version.sizeBytes)
                        : Collections.emptyMap())
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

    private Map<tech.kayys.alkhawarizm.core.model.ModelFormat, tech.kayys.alkhawarizm.spi.model.ArtifactLocation> buildArtifacts(
            String format,
            String storageUri,
            String checksum,
            Long sizeBytes) {
        if (storageUri == null || storageUri.isBlank()) {
            return Collections.emptyMap();
        }

        tech.kayys.alkhawarizm.core.model.ModelFormat resolved = resolveFormat(format, storageUri);
        if (resolved == null) {
            return Collections.emptyMap();
        }

        return Map.of(resolved,
                new tech.kayys.alkhawarizm.spi.model.ArtifactLocation(
                        storageUri,
                        checksum,
                        sizeBytes,
                        null));
    }

    private tech.kayys.alkhawarizm.core.model.ModelFormat resolveFormat(String format, String storageUri) {
        if (format != null && !format.isBlank()) {
            String normalized = format.trim().toLowerCase();
            if ("litert".equals(normalized) || "litert".equals(normalized)) {
                return tech.kayys.alkhawarizm.core.model.ModelFormat.LITERT;
            }
            return tech.kayys.alkhawarizm.core.model.ModelFormat.findByRuntime(format)
                    .or(() -> tech.kayys.alkhawarizm.core.model.ModelFormat.findByExtension(format))
                    .orElse(null);
        }

        if (storageUri != null) {
            String path = storageUri;
            int q = path.indexOf('?');
            if (q >= 0) {
                path = path.substring(0, q);
            }
            int dot = path.lastIndexOf('.');
            if (dot >= 0 && dot + 1 < path.length()) {
                String ext = path.substring(dot + 1);
                return tech.kayys.alkhawarizm.core.model.ModelFormat.findByExtension(ext).orElse(null);
            }
        }

        return null;
    }

    private String resolveRequestId(RequestContext requestContext) {
        if (requestContext == null || requestContext.getRequestId() == null) {
            return "community";
        }
        return resolveRequestId(requestContext.getRequestId());
    }

    private String resolveRequestId(String requestId) {
        return (requestId == null || requestId.isBlank()) ? "community" : requestId.trim();
    }
}
