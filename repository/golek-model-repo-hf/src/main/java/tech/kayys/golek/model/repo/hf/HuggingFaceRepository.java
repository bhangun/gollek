package tech.kayys.golek.model.repository.hf;

import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

import tech.kayys.golek.model.core.ModelArtifact;
import tech.kayys.golek.model.core.ModelDescriptor;
import tech.kayys.golek.model.core.ModelRef;
import tech.kayys.golek.model.core.ModelRepository;

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
        return "hf".equalsIgnoreCase(ref.scheme());
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
                        "filename", artifact.filename()));
    }

    @Override
    public ModelArtifact fetch(ModelDescriptor descriptor) {
        Path target = cacheDir
                .resolve("hf")
                .resolve(descriptor.id().replace("/", "_").replace(":", "_"));

        return downloader.download(descriptor, target);
    }
}
