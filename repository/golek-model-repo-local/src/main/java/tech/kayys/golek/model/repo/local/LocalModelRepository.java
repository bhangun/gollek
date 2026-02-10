package tech.kayys.golek.model.repo.local;

import jakarta.enterprise.context.ApplicationScoped;

import io.smallrye.mutiny.Uni;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.Map;
import java.util.List;
import java.util.Collections;

import tech.kayys.golek.model.core.ModelRepository;
import tech.kayys.golek.spi.model.ModelArtifact;
import tech.kayys.golek.spi.model.ModelDescriptor;
import tech.kayys.golek.spi.model.ModelRef;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.Pageable;

@ApplicationScoped
public final class LocalModelRepository implements ModelRepository {

    @Override
    public boolean supports(ModelRef ref) {
        return "local".equalsIgnoreCase(ref.scheme());
    }

    @Override
    public ModelDescriptor resolve(ModelRef ref) {
        String path = ref.parameters().get("path");
        return new ModelDescriptor(
                ref.name(),
                ref.parameters().getOrDefault("format", "auto"),
                path != null ? URI.create("file://" + path) : URI.create("file://"),
                Map.of("provider", "local"));
    }

    @Override
    public ModelArtifact fetch(ModelDescriptor descriptor) {
        Path path = Path.of(descriptor.source());
        return new ModelArtifact(path, "local", descriptor.metadata());
    }

    @Override
    public Uni<ModelManifest> findById(String modelId, String tenantId) {
        if (modelId == null)
            return Uni.createFrom().nullItem();

        // Normalize modelId (replace / with _)
        String normalizedId = modelId.replace("/", "_");

        Path modelDir = Path.of(System.getProperty("user.home"), ".golek", "models", "gguf");
        Path modelPath = modelDir.resolve(normalizedId);

        // Check local variations (e.g. if the file has -GGUF suffix)
        if (!Files.exists(modelPath)) {
            if (Files.exists(modelDir.resolve(normalizedId + "-GGUF"))) {
                modelPath = modelDir.resolve(normalizedId + "-GGUF");
            } else if (normalizedId.endsWith("_GGUF")
                    && Files.exists(modelDir.resolve(normalizedId.substring(0, normalizedId.length() - 5)))) {
                modelPath = modelDir.resolve(normalizedId.substring(0, normalizedId.length() - 5));
            }
        }

        if (Files.exists(modelPath)) {
            try {
                long size = Files.size(modelPath);
                String uri = modelPath.toUri().toString();

                tech.kayys.golek.spi.model.ArtifactLocation artifact = new tech.kayys.golek.spi.model.ArtifactLocation(
                        uri, null, size, "application/octet-stream");

                return Uni.createFrom().item(ModelManifest.builder()
                        .modelId(modelId)
                        .name(modelId)
                        .version("1.0.0")
                        .tenantId(tenantId)
                        .artifacts(Map.of(ModelFormat.GGUF, artifact))
                        .metadata(Map.of("path", uri, "source", "local"))
                        .build());
            } catch (Exception e) {
                // Fallback if file read fails
                return Uni.createFrom().item(ModelManifest.builder()
                        .modelId(modelId)
                        .name(modelId)
                        .version("1.0.0")
                        .tenantId(tenantId)
                        .build());
            }
        }
        return Uni.createFrom().nullItem();
    }

    @Override
    public Uni<List<ModelManifest>> list(String tenantId, Pageable pageable) {
        // Just return empty for now, findById is more important for the 'run' command
        return Uni.createFrom().item(Collections.emptyList());
    }

    @Override
    public Uni<ModelManifest> save(ModelManifest manifest) {
        // No-op for now, file existence is our "database"
        return Uni.createFrom().item(manifest);
    }

    @Override
    public Uni<Void> delete(String modelId, String tenantId) {
        return Uni.createFrom().nullItem();
    }

    @Override
    public Path downloadArtifact(ModelManifest manifest, ModelFormat format) {
        return null;
    }

    @Override
    public boolean isCached(String modelId, ModelFormat format) {
        return true; // Local is "always cached"
    }

    @Override
    public void evictCache(String modelId, ModelFormat format) {
        // No-op for local
    }
}