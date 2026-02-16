package tech.kayys.golek.model.repo.local;

import jakarta.enterprise.context.ApplicationScoped;

import io.smallrye.mutiny.Uni;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.ArrayList;
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

        Path modelPath = resolveLocalPath(modelId);
        Path artifactPath = resolveArtifactFile(modelPath);

        if (artifactPath != null && Files.exists(artifactPath)) {
            try {
                long size = Files.size(artifactPath);
                String uri = artifactPath.toUri().toString();
                ModelFormat format = detectFormat(artifactPath);

                tech.kayys.golek.spi.model.ArtifactLocation artifact = new tech.kayys.golek.spi.model.ArtifactLocation(
                        uri, null, size, "application/octet-stream");

                return Uni.createFrom().item(ModelManifest.builder()
                        .modelId(modelId)
                        .name(modelId)
                        .version("1.0.0")
                        .requestId("local-" + modelId.replace("/", "_"))
                        .path(uri)
                        .apiKey(tenantId != null && !tenantId.isBlank() ? tenantId : "community")
                        .artifacts(Map.of(format, artifact))
                        .metadata(Map.of("path", uri, "source", "local", "format", format.name()))
                        .build());
            } catch (Exception e) {
                // Fallback if file read fails
                return Uni.createFrom().item(ModelManifest.builder()
                        .modelId(modelId)
                        .name(modelId)
                        .version("1.0.0")
                        .requestId("local-" + modelId.replace("/", "_"))
                        .path(artifactPath.toString())
                        .apiKey(tenantId != null && !tenantId.isBlank() ? tenantId : "community")
                        .build());
            }
        }
        return Uni.createFrom().nullItem();
    }

    private Path resolveLocalPath(String modelId) {
        Path direct = Path.of(modelId);
        if (direct.isAbsolute() && Files.exists(direct)) {
            return direct;
        }

        String normalizedId = modelId.replace("/", "_");
        Path ggufDir = Path.of(System.getProperty("user.home"), ".golek", "models", "gguf");
        Path torchDir = Path.of(System.getProperty("user.home"), ".golek", "models", "torchscript");

        List<Path> candidates = new ArrayList<>();

        // GGUF variants (historical naming)
        candidates.add(ggufDir.resolve(normalizedId));
        candidates.add(ggufDir.resolve(normalizedId + ".gguf"));
        candidates.add(ggufDir.resolve(normalizedId + "-GGUF"));
        candidates.add(ggufDir.resolve(modelId + ".gguf"));

        // LibTorch / PyTorch variants (both nested repo path and normalized file naming)
        candidates.add(torchDir.resolve(modelId));
        candidates.add(torchDir.resolve(normalizedId));
        for (String ext : List.of(".pt", ".pts", ".pth", ".bin", ".safetensors")) {
            candidates.add(torchDir.resolve(modelId + ext));
            candidates.add(torchDir.resolve(normalizedId + ext));
        }

        for (Path candidate : candidates) {
            if (Files.exists(candidate)) {
                return candidate;
            }
        }
        // Return primary GGUF candidate for consistent "not found" behavior.
        return ggufDir.resolve(normalizedId);
    }

    private Path resolveArtifactFile(Path modelPath) {
        if (modelPath == null || !Files.exists(modelPath)) {
            return null;
        }
        if (Files.isRegularFile(modelPath)) {
            return modelPath;
        }
        if (!Files.isDirectory(modelPath)) {
            return null;
        }
        try (var stream = Files.list(modelPath)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(this::isLikelyModelFile)
                    .findFirst()
                    .orElse(null);
        } catch (Exception ignored) {
            return null;
        }
    }

    private boolean isLikelyModelFile(Path p) {
        String name = p.getFileName().toString().toLowerCase();
        return name.endsWith(".gguf")
                || name.endsWith(".pt")
                || name.endsWith(".pts")
                || name.endsWith(".pth")
                || name.endsWith(".bin")
                || name.endsWith(".safetensors")
                || isGgufFile(p);
    }

    private ModelFormat detectFormat(Path modelPath) {
        String name = modelPath.getFileName().toString().toLowerCase();
        if (name.endsWith(".gguf")) {
            return ModelFormat.GGUF;
        }
        // Handle extensionless GGUF files by checking magic header.
        if (isGgufFile(modelPath)) {
            return ModelFormat.GGUF;
        }
        if (name.endsWith(".pt") || name.endsWith(".pts")) {
            return ModelFormat.TORCHSCRIPT;
        }
        if (name.endsWith(".pth") || name.endsWith(".bin") || name.endsWith(".safetensors")) {
            return ModelFormat.PYTORCH;
        }
        return ModelFormat.UNKNOWN;
    }

    private boolean isGgufFile(Path modelPath) {
        try {
            if (!Files.isRegularFile(modelPath) || Files.size(modelPath) < 4) {
                return false;
            }
            byte[] header = new byte[4];
            try (var in = Files.newInputStream(modelPath)) {
                int read = in.read(header);
                if (read < 4) {
                    return false;
                }
            }
            return header[0] == 'G' && header[1] == 'G' && header[2] == 'U' && header[3] == 'F';
        } catch (Exception ignored) {
            return false;
        }
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
