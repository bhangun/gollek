package tech.kayys.gollek.model.repo.hf;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.Collections;
import java.util.Locale;
import java.util.Optional;
import org.jboss.logging.Logger;
import tech.kayys.gollek.model.download.DownloadProgressListener;

import tech.kayys.gollek.model.core.ModelRepository;
import tech.kayys.gollek.spi.model.ArtifactLocation;
import tech.kayys.gollek.spi.model.ModelArtifact;
import tech.kayys.gollek.spi.model.ModelDescriptor;
import tech.kayys.gollek.spi.model.ModelRef;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.Pageable;

@ApplicationScoped
public final class HuggingFaceRepository implements ModelRepository {

    private static final Logger LOG = Logger.getLogger(HuggingFaceRepository.class);
    private static final String DEFAULT_REVISION = "main";

    private final HuggingFaceConfig config;
    private final HuggingFaceClient client;
    private final HuggingFaceArtifactResolver resolver;
    private final HuggingFaceDownloader downloader;
    private final Path cacheDir;

    @Inject
    public HuggingFaceRepository(HuggingFaceClient client, HuggingFaceConfig config) {
        this(resolveDefaultCacheDir(), client, config);
    }

    public HuggingFaceRepository(Path cacheDir, HuggingFaceClient client) {
        this(cacheDir, client, null);
    }

    public HuggingFaceRepository(Path cacheDir, HuggingFaceClient client, HuggingFaceConfig config) {
        this.cacheDir = cacheDir;
        this.client = client;
        this.config = config;
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
        if (!isHuggingFaceModelId(modelId)) {
            return Uni.createFrom().nullItem();
        }
        if (config != null && !config.autoDownload()) {
            return Uni.createFrom().nullItem();
        }

        return Uni.createFrom().item(() -> {
            try {
                String repoId = normalizeRepoId(modelId);
                Path modelDir = cacheDir.resolve("safetensors").resolve(repoId);
                Files.createDirectories(modelDir);

                Path preferredArtifact = findBestLocalArtifact(modelDir);
                if (preferredArtifact == null) {
                    preferredArtifact = downloadBestArtifact(repoId, modelDir);
                } else {
                    ensureLocalSidecars(repoId, modelDir);
                }

                if (preferredArtifact == null || !Files.exists(preferredArtifact)) {
                    return null;
                }

                ModelFormat format = detectFormat(preferredArtifact);
                long size = Files.size(preferredArtifact);
                String uri = preferredArtifact.toUri().toString();

                return ModelManifest.builder()
                        .modelId(preferredArtifact.toString())
                        .name(repoId)
                        .version(DEFAULT_REVISION)
                        .requestId(requestId != null && !requestId.isBlank() ? requestId : "community")
                        .path(preferredArtifact.toString())
                        .apiKey(requestId != null && !requestId.isBlank() ? requestId : "community")
                        .artifacts(Map.of(format, new ArtifactLocation(uri, null, size, "application/octet-stream")))
                        .metadata(Map.of(
                                "source", "huggingface",
                                "repo", repoId,
                                "path", preferredArtifact.toString(),
                                "format", format.name()))
                        .build();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }).onFailure().invoke(e -> LOG.warnf("HF auto-download failed for %s: %s", modelId, e.getMessage()))
                .onFailure().recoverWithNull();
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
        if (manifest == null) {
            return null;
        }
        String modelId = manifest.modelId();
        Path asPath = Path.of(modelId);
        if (Files.exists(asPath)) {
            return asPath;
        }
        return null;
    }

    @Override
    public boolean isCached(String modelId, ModelFormat format) {
        if (!isHuggingFaceModelId(modelId)) {
            return false;
        }
        String repoId = normalizeRepoId(modelId);
        Path modelDir = cacheDir.resolve("safetensors").resolve(repoId);
        Path artifact = findBestLocalArtifact(modelDir);
        return artifact != null && Files.exists(artifact);
    }

    @Override
    public void evictCache(String modelId, ModelFormat format) {
        // No-op
    }

    private static Path resolveDefaultCacheDir() {
        return Path.of(System.getProperty("user.home"), ".gollek", "models");
    }

    private boolean isHuggingFaceModelId(String modelId) {
        return modelId != null && (modelId.startsWith("hf:") || modelId.contains("/"));
    }

    private String normalizeRepoId(String modelId) {
        if (modelId == null) {
            return "";
        }
        String normalized = modelId.trim();
        if (normalized.startsWith("hf:")) {
            normalized = normalized.substring(3);
        }
        return normalized;
    }

    private Path downloadBestArtifact(String repoId, Path targetDir) throws Exception {
        List<String> files = client.listFiles(repoId);
        Optional<String> exact = files.stream()
                .filter(name -> "model.safetensors".equalsIgnoreCase(name) || "model.safetensor".equalsIgnoreCase(name))
                .findFirst();

        if (exact.isPresent()) {
            Path target = targetDir.resolve(fileNameOnly(exact.get()));
            if (!Files.exists(target)) {
                client.downloadFile(repoId, exact.get(), target, progressPrinter(repoId, exact.get()));
            }
            downloadSidecars(repoId, targetDir, files);
            validateDownloadedArtifact(target);
            return target;
        }

        Optional<String> genericSafetensors = files.stream()
                .filter(this::isSafetensorFile)
                .findFirst();

        if (genericSafetensors.isPresent()) {
            Path target = targetDir.resolve(fileNameOnly(genericSafetensors.get()));
            if (!Files.exists(target)) {
                client.downloadFile(repoId, genericSafetensors.get(), target, progressPrinter(repoId, genericSafetensors.get()));
            }
            downloadSidecars(repoId, targetDir, files);
            validateDownloadedArtifact(target);
            return target;
        }

        return null;
    }

    private Path findBestLocalArtifact(Path modelDir) {
        if (modelDir == null || !Files.exists(modelDir)) {
            return null;
        }
        try (var files = Files.list(modelDir)) {
            return files
                    .filter(Files::isRegularFile)
                    .filter(this::isSafetensorFile)
                    .sorted((a, b) -> Integer.compare(priority(b.getFileName().toString()),
                            priority(a.getFileName().toString())))
                    .findFirst()
                    .orElse(null);
        } catch (Exception e) {
            return null;
        }
    }

    private boolean isSafetensorFile(Path path) {
        return isSafetensorFile(path.getFileName().toString());
    }

    private boolean isSafetensorFile(String name) {
        if (name == null) {
            return false;
        }
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".safetensors") || lower.endsWith(".safetensor");
    }

    private int priority(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        if (lower.equals("model.safetensors") || lower.equals("model.safetensor")) {
            return 100;
        }
        if (lower.endsWith(".safetensors") || lower.endsWith(".safetensor")) {
            return 50;
        }
        return 0;
    }

    private String fileNameOnly(String remotePath) {
        int idx = remotePath.lastIndexOf('/');
        return idx >= 0 ? remotePath.substring(idx + 1) : remotePath;
    }

    private ModelFormat detectFormat(Path artifactPath) {
        String name = artifactPath.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
            return ModelFormat.SAFETENSORS;
        }
        return ModelFormat.PYTORCH;
    }

    private void validateDownloadedArtifact(Path artifact) throws java.io.IOException {
        if (artifact == null || !Files.exists(artifact) || !Files.isRegularFile(artifact)) {
            throw new java.io.IOException("Downloaded artifact is missing: " + artifact);
        }
        long size = Files.size(artifact);
        if (size <= 0) {
            throw new java.io.IOException("Downloaded artifact is empty: " + artifact);
        }
    }

    private DownloadProgressListener progressPrinter(String repoId, String filename) {
        return new DownloadProgressListener() {
            @Override
            public void onStart(long totalBytes) {
                LOG.infof("Downloading %s from %s...", filename, repoId);
            }

            @Override
            public void onProgress(long downloadedBytes, long totalBytes, double progress) {
                if (totalBytes > 0) {
                    int percent = (int) Math.min(100, Math.round(progress * 100));
                    System.out.printf("\rHF download %s: %d%% (%d/%d MB)",
                            filename,
                            percent,
                            downloadedBytes / 1024 / 1024,
                            totalBytes / 1024 / 1024);
                } else {
                    System.out.printf("\rHF download %s: %d MB", filename, downloadedBytes / 1024 / 1024);
                }
            }

            @Override
            public void onComplete(long totalBytes) {
                System.out.println();
                LOG.infof("Download complete: %s (%d MB)", filename, totalBytes / 1024 / 1024);
            }

            @Override
            public void onError(Throwable error) {
                System.out.println();
                LOG.warnf("Download failed for %s/%s: %s", repoId, filename,
                        error != null ? error.getMessage() : "unknown error");
            }
        };
    }

    private void downloadSidecars(String repoId, Path targetDir, List<String> availableFiles) {
        for (String sidecar : sidecarCandidates()) {
            Optional<String> remote = availableFiles.stream()
                    .filter(name -> name.equalsIgnoreCase(sidecar))
                    .findFirst();
            if (remote.isEmpty()) {
                continue;
            }
            Path target = targetDir.resolve(fileNameOnly(remote.get()));
            if (Files.exists(target)) {
                continue;
            }
            try {
                client.downloadFile(repoId, remote.get(), target, progressPrinter(repoId, remote.get()));
            } catch (Exception e) {
                LOG.warnf("Failed to download sidecar %s for %s: %s", remote.get(), repoId, e.getMessage());
            }
        }
    }

    private void ensureLocalSidecars(String repoId, Path targetDir) {
        try {
            List<String> files = client.listFiles(repoId);
            downloadSidecars(repoId, targetDir, files);
        } catch (Exception e) {
            LOG.debugf("Unable to refresh sidecars for %s: %s", repoId, e.getMessage());
        }
    }

    private List<String> sidecarCandidates() {
        return List.of(
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "special_tokens_map.json",
                "processor_config.json",
                "preprocessor_config.json");
    }
}
