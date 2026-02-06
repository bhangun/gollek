package tech.kayys.golek.model.core;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.model.download.DownloadProgressListener;

import java.nio.file.Path;
import java.util.Optional;

/**
 * Service to synchronize models between remote and local repositories
 */
@ApplicationScoped
public class ModelSyncService {

    private static final Logger LOG = Logger.getLogger(ModelSyncService.class);

    @Inject
    LocalModelRepository localRepo;

    @Inject
    Instance<RemoteModelRepository> remoteRepos;

    /**
     * Sync model from remote to local
     */
    public Uni<ModelManifest> sync(String modelId, String tenantId, String remoteType,
            DownloadProgressListener listener) {
        RemoteModelRepository remoteRepo = findRemoteRepo(remoteType)
                .orElseThrow(() -> new IllegalArgumentException("Unsupported remote repository type: " + remoteType));

        LOG.infof("Starting sync for model %s from %s", modelId, remoteType);

        return remoteRepo.fetchMetadata(modelId, tenantId)
                .flatMap(manifest -> {
                    // Start download
                    // For now, we assume we download all artifacts in the manifest
                    // In a more advanced impl, we might filter by format/device
                    return downloadAllArtifacts(remoteRepo, manifest, listener)
                            .replaceWith(manifest);
                })
                .flatMap(manifest -> localRepo.save(manifest))
                .onItem().invoke(m -> LOG.infof("Successfully synced model %s", m.modelId()));
    }

    private Uni<Void> downloadAllArtifacts(RemoteModelRepository remoteRepo, ModelManifest manifest,
            DownloadProgressListener listener) {
        // Simple implementation: download sequentially for now
        // Advanced: parallel downloads
        Uni<Void> chain = Uni.createFrom().nullItem();

        for (String artifactId : manifest.artifacts().keySet().stream().map(Enum::name).toList()) {
            chain = chain
                    .flatMap(v -> remoteRepo.downloadArtifact(manifest, artifactId, null, listener).replaceWithVoid());
        }

        return chain;
    }

    private Optional<RemoteModelRepository> findRemoteRepo(String type) {
        return remoteRepos.stream()
                .filter(r -> r.type().equalsIgnoreCase(type))
                .findFirst();
    }
}
