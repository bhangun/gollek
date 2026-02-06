package tech.kayys.golek.provider.huggingface;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import tech.kayys.golek.spi.model.ArtifactLocation;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.model.core.RemoteModelRepository;
import tech.kayys.golek.model.download.DownloadProgressListener;

import java.nio.file.Path;
import java.time.Instant;
import java.util.*;

/**
 * HuggingFace implementation of RemoteModelRepository
 */
@ApplicationScoped
public class HuggingFaceModelRepository implements RemoteModelRepository {

    @Inject
    HuggingFaceClient client;

    @Inject
    tech.kayys.golek.model.download.DownloadManager downloadManager;

    @ConfigProperty(name = "hf.api.token")
    Optional<String> apiToken;

    @Override
    public String type() {
        return "huggingface";
    }

    @Override
    public Uni<ModelManifest> fetchMetadata(String modelId, String tenantId) {
        return Uni.createFrom().item(() -> {
            try {
                Map<String, Object> metadata = client.getModelMetadata(modelId, apiToken);
                return mapToManifest(modelId, tenantId, metadata);
            } catch (Exception e) {
                throw new RuntimeException("Failed to fetch HF metadata", e);
            }
        });
    }

    @Override
    public Uni<List<ModelManifest>> search(String query, String tenantId) {
        // Implement HF search if needed
        return Uni.createFrom().item(List.of());
    }

    @Override
    public Uni<Path> downloadArtifact(ModelManifest manifest, String artifactId, Path targetDir,
            DownloadProgressListener listener) {
        return Uni.createFrom().item(() -> {
            try {
                String filename = artifactId;
                Path targetPath = targetDir.resolve(filename);

                // Use parallel downloader
                return downloadManager.downloadParallel(
                        manifest.modelId(),
                        targetPath,
                        -1L, // Total bytes unknown, or fetch metadata first
                        (start, end) -> {
                            try {
                                return client.downloadRange(manifest.modelId(), filename, start, end, apiToken);
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        },
                        listener).toCompletableFuture().get();
            } catch (Exception e) {
                throw new RuntimeException("Failed to download HF artifact", e);
            }
        });
    }

    private ModelManifest mapToManifest(String modelId, String tenantId, Map<String, Object> metadata) {
        Map<ModelFormat, ArtifactLocation> artifacts = new HashMap<>();

        // Detection logic for formats based on files (siblings)
        List<Map<String, Object>> siblings = (List<Map<String, Object>>) metadata.get("siblings");
        if (siblings != null) {
            for (Map<String, Object> sibling : siblings) {
                String rfilename = (String) sibling.get("rfilename");
                ModelFormat.findByExtension(getExtension(rfilename)).ifPresent(format -> {
                    artifacts.put(format, new ArtifactLocation(
                            "hf://" + modelId + "/" + rfilename,
                            null, // Checksum not always direct in this API
                            null, // Size from metadata if available
                            null));
                });
            }
        }

        return new ModelManifest(
                modelId,
                (String) metadata.get("id"),
                "latest", // HF usually doesn't have semver versions in the same way
                tenantId,
                artifacts,
                List.of(), // Devices to be detected by backend
                null, // Resource requirements
                metadata,
                Instant.now(),
                Instant.now());
    }

    private String getExtension(String filename) {
        int lastDot = filename.lastIndexOf('.');
        return lastDot > 0 ? filename.substring(lastDot + 1) : "";
    }
}
