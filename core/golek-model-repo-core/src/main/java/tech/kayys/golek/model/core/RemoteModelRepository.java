package tech.kayys.golek.model.core;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.model.download.DownloadProgressListener;

import java.nio.file.Path;
import java.util.List;

/**
 * Interface for remote model repositories (e.g., HuggingFace, S3)
 */
public interface RemoteModelRepository {

    String type();

    Uni<ModelManifest> fetchMetadata(String modelId, String tenantId);

    Uni<List<ModelManifest>> search(String query, String tenantId);

    Uni<Path> downloadArtifact(ModelManifest manifest, String artifactId, Path targetDir,
            DownloadProgressListener listener);
}
