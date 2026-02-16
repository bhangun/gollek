package tech.kayys.golek.model.repo.hf;

import io.smallrye.mutiny.Uni;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Collections;

import tech.kayys.golek.model.core.ModelRepository;
import tech.kayys.golek.spi.model.ModelArtifact;
import tech.kayys.golek.spi.model.ModelDescriptor;
import tech.kayys.golek.spi.model.ModelRef;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.Pageable;

public final class HuggingFaceRepository implements ModelRepository {

    private final HuggingFaceClient client;
    private final HuggingFaceArtifactResolver resolver;
    private final HuggingFaceDownloader downloader;
    private final Path cacheDir;

    public HuggingFaceRepository(Path cacheDir, HuggingFaceClient client) {
        this.cacheDir = cacheDir;
        this.client = client;
        this.resolver = new HuggingFaceArtifactResolver(client);
        this.downloader = new HuggingFaceDownloader(client);
    }

    @Override
    public boolean supports(ModelRef ref) {
        return "hf".equalsIgnoreCase(ref.scheme()) || "huggingface".equalsIgnoreCase(ref.scheme());
    }

    @Override
    public ModelDescriptor resolve(ModelRef ref) {
        HuggingFaceArtifact artifact = resolver.resolve(ref);

        return new ModelDescriptor(
                artifact.id(),
                artifact.format(),
                artifact.downloadUri(),
                Map.of(
                        "provider", "huggingface",
                        "repo", artifact.repo(),
                        "revision", artifact.revision(),
                        "filename", artifact.filename() != null ? artifact.filename() : ""));
    }

    @Override
    public ModelArtifact fetch(ModelDescriptor descriptor) {
        Path target = cacheDir
                .resolve("hf")
                .resolve(descriptor.id().replace("/", "_").replace(":", "_"));

        return downloader.download(descriptor, target);
    }

    @Override
    public Uni<ModelManifest> findById(String modelId, String requestId) {
        return Uni.createFrom().failure(new UnsupportedOperationException("HF findById not yet implemented"));
    }

    @Override
    public Uni<List<ModelManifest>> list(String requestId, Pageable pageable) {
        return Uni.createFrom().item(Collections.emptyList());
    }

    @Override
    public Uni<ModelManifest> save(ModelManifest manifest) {
        return Uni.createFrom().failure(new UnsupportedOperationException("Cannot save to HuggingFace repository"));
    }

    @Override
    public Uni<Void> delete(String modelId, String requestId) {
        return Uni.createFrom().failure(new UnsupportedOperationException("Cannot delete from HuggingFace repository"));
    }

    @Override
    public Path downloadArtifact(ModelManifest manifest, ModelFormat format) {
        throw new UnsupportedOperationException("HF downloadArtifact not yet implemented");
    }

    @Override
    public boolean isCached(String modelId, ModelFormat format) {
        return false;
    }

    @Override
    public void evictCache(String modelId, ModelFormat format) {
        // No-op
    }
}
