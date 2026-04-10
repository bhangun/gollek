package tech.kayys.gollek.model.repo.kaggle;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
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

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Kaggle model repository implementation.
 *
 * Supports downloading models from kaggle.com/models in GGUF,
 * SafeTensors, and LiteRT formats.
 */
@ApplicationScoped
public final class KaggleRepository implements ModelRepository {

    private static final Logger LOG = Logger.getLogger(KaggleRepository.class);
    private static final String DEFAULT_REVISION = "main";

    private final KaggleConfig config;
    private final KaggleClient client;
    private final KaggleArtifactResolver resolver;
    private final Path cacheDir;

    @Inject
    public KaggleRepository(KaggleClient client, KaggleConfig config) {
        this(resolveDefaultCacheDir(), client, config);
    }

    public KaggleRepository(Path cacheDir, KaggleClient client) {
        this(cacheDir, client, null);
    }

    public KaggleRepository(Path cacheDir, KaggleClient client, KaggleConfig config) {
        this.cacheDir = cacheDir;
        this.client = client;
        this.config = config;
        this.resolver = new KaggleArtifactResolver(client);
    }

    @Override
    public boolean supports(ModelRef ref) {
        return "kaggle".equalsIgnoreCase(ref.scheme());
    }

    @Override
    public ModelDescriptor resolve(ModelRef ref) {
        KaggleArtifactResolver.KaggleArtifact artifact = resolver.resolve(ref);
        String format = artifact.format();
        return new ModelDescriptor(
                artifact.slug(),
                format,
                URI.create("kaggle://" + artifact.slug()),
                Map.of(
                        "provider", "kaggle",
                        "slug", artifact.slug(),
                        "revision", artifact.revision(),
                        "format", artifact.format()));
    }

    @Override
    public ModelArtifact fetch(ModelDescriptor descriptor) {
        Path target = resolveFormatSpecificCacheDir(descriptor);
        try {
            String slug = descriptor.id();
            ModelFormat fmt = ModelFormat.valueOf(descriptor.format().toUpperCase(Locale.ROOT));
            List<String> files = client.listFiles(slug);

            Optional<String> bestFile = pickBestFile(files, fmt);
            if (bestFile.isEmpty()) {
                throw new IOException("No suitable model file found in " + slug);
            }

            Path targetFile = target.resolve(fileNameOnly(bestFile.get()));
            Files.createDirectories(targetFile.getParent());

            client.downloadFile(slug, bestFile.get(), targetFile, progressPrinter(slug, bestFile.get()));
            downloadSidecars(slug, target, files);

            long size = Files.size(targetFile);
            return new ModelArtifact(targetFile, null, Map.of("format", descriptor.format(), "size", String.valueOf(size)));
        } catch (Exception e) {
            throw new RuntimeException("Failed to fetch Kaggle model: " + descriptor.id(), e);
        }
    }

    @Override
    public Uni<ModelManifest> findById(String modelId, String requestId) {
        if (!isKaggleModelId(modelId)) {
            return Uni.createFrom().nullItem();
        }
        if (config != null && !config.autoDownload()) {
            return Uni.createFrom().nullItem();
        }

        return Uni.createFrom().item(() -> {
            try {
                String slug = normalizeSlug(modelId);
                Path modelDir = cacheDir.resolve("kaggle").resolve(slug.replace("/", "_"));

                Map<ModelFormat, ArtifactLocation> artifacts = discoverLocalArtifacts(modelDir);

                if (artifacts.isEmpty() && isAutoDownloadEnabled()) {
                    try {
                        List<String> files = client.listFiles(slug);
                        Files.createDirectories(modelDir);
                        Optional<String> bestFile = pickBestFile(files, null);
                        if (bestFile.isPresent()) {
                            Path targetFile = modelDir.resolve(fileNameOnly(bestFile.get()));
                            client.downloadFile(slug, bestFile.get(), targetFile,
                                    progressPrinter(slug, bestFile.get()));
                            downloadSidecars(slug, modelDir, files);
                            artifacts = discoverLocalArtifacts(modelDir);
                        }
                    } catch (Exception e) {
                        LOG.warnf("Kaggle download failed for %s: %s", slug, e.getMessage());
                    }
                }

                if (artifacts.isEmpty()) {
                    return null;
                }

                Path primaryPath = selectPrimaryPath(artifacts);
                String primaryFormat = selectPrimaryFormat(artifacts);

                return ModelManifest.builder()
                        .modelId(primaryPath.toString())
                        .name(slug)
                        .version(DEFAULT_REVISION)
                        .requestId(requestId != null && !requestId.isBlank() ? requestId : "community")
                        .path(primaryPath.toString())
                        .apiKey(requestId != null && !requestId.isBlank() ? requestId : "community")
                        .artifacts(artifacts)
                        .metadata(Map.of(
                                "source", "kaggle",
                                "slug", slug,
                                "primary_format", primaryFormat))
                        .build();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }).onFailure().invoke(e -> LOG.warnf("Kaggle auto-download failed for %s: %s", modelId, e.getMessage()))
                .onFailure().recoverWithNull();
    }

    @Override
    public Uni<List<ModelManifest>> list(String requestId, Pageable pageable) {
        return Uni.createFrom().item(Collections.emptyList());
    }

    @Override
    public Uni<ModelManifest> save(ModelManifest manifest) {
        return Uni.createFrom().failure(new UnsupportedOperationException("Cannot save to Kaggle repository"));
    }

    @Override
    public Uni<Void> delete(String modelId, String requestId) {
        return Uni.createFrom().failure(new UnsupportedOperationException("Cannot delete from Kaggle repository"));
    }

    @Override
    public Path downloadArtifact(ModelManifest manifest, ModelFormat format) {
        if (manifest == null) return null;
        Path asPath = Path.of(manifest.modelId());
        return Files.exists(asPath) ? asPath : null;
    }

    @Override
    public boolean isCached(String modelId, ModelFormat format) {
        if (!isKaggleModelId(modelId)) return false;
        String slug = normalizeSlug(modelId);
        Path modelDir = cacheDir.resolve("kaggle").resolve(slug.replace("/", "_"));
        return findBestLocalArtifact(modelDir, format) != null;
    }

    @Override
    public void evictCache(String modelId, ModelFormat format) {
        // No-op
    }

    // ── Private helpers ──────────────────────────────────────────────────

    private static Path resolveDefaultCacheDir() {
        return Path.of(System.getProperty("user.home"), ".gollek", "models");
    }

    private boolean isKaggleModelId(String modelId) {
        return modelId != null && (modelId.startsWith("kaggle:") || modelId.contains("/"));
    }

    private String normalizeSlug(String modelId) {
        if (modelId == null) return "";
        String normalized = modelId.trim();
        if (normalized.startsWith("kaggle:")) normalized = normalized.substring(7);
        return normalized;
    }

    private Path resolveFormatSpecificCacheDir(ModelDescriptor descriptor) {
        String slug = descriptor.id().replace("/", "_");
        String format = descriptor.format().toLowerCase(Locale.ROOT);
        return cacheDir.resolve("kaggle").resolve(format).resolve(slug);
    }

    private Map<ModelFormat, ArtifactLocation> discoverLocalArtifacts(Path modelDir) {
        Map<ModelFormat, ArtifactLocation> artifacts = new HashMap<>();
        if (!Files.exists(modelDir)) return artifacts;

        try (var stream = Files.list(modelDir)) {
            stream.filter(Files::isRegularFile).forEach(p -> {
                String name = p.getFileName().toString();
                ModelFormat fmt = detectFormat(name);
                if (fmt != null) {
                    try {
                        artifacts.put(fmt, new ArtifactLocation(
                                p.toUri().toString(), null, Files.size(p), "application/octet-stream"));
                    } catch (IOException ignored) {
                    }
                }
            });
        } catch (IOException ignored) {
        }
        return artifacts;
    }

    private Optional<String> pickBestFile(List<String> files, ModelFormat preferred) {
        if (files.isEmpty()) return Optional.empty();

        if (preferred != null) {
            return files.stream()
                    .filter(f -> formatMatches(f, preferred))
                    .min((a, b) -> priority(b) - priority(a));
        }

        return files.stream()
                .filter(f -> isGgufFile(f) || isSafetensorFile(f) || isPytorchFile(f) || isLitertFile(f))
                .min((a, b) -> priority(b) - priority(a))
                .or(() -> files.stream().findFirst());
    }

    private boolean formatMatches(String filename, ModelFormat format) {
        return switch (format) {
            case GGUF -> isGgufFile(filename);
            case SAFETENSORS -> isSafetensorFile(filename);
            case PYTORCH -> isPytorchFile(filename);
            case LITERT -> isLitertFile(filename);
            default -> false;
        };
    }

    private int priority(String filename) {
        String lower = filename.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".gguf")) return 100;
        if (lower.equals("model.safetensors")) return 90;
        if (lower.endsWith(".safetensors")) return 80;
        if (lower.equals("pytorch_model.bin")) return 70;
        if (lower.endsWith(".bin")) return 60;
        if (lower.endsWith(".litertlm")) return 50;
        return 0;
    }

    private ModelFormat detectFormat(String filename) {
        if (isGgufFile(filename)) return ModelFormat.GGUF;
        if (isSafetensorFile(filename)) return ModelFormat.SAFETENSORS;
        if (isPytorchFile(filename)) return ModelFormat.PYTORCH;
        if (isLitertFile(filename)) return ModelFormat.LITERT;
        return null;
    }

    private Path selectPrimaryPath(Map<ModelFormat, ArtifactLocation> artifacts) {
        if (artifacts.containsKey(ModelFormat.GGUF))
            return Path.of(artifacts.get(ModelFormat.GGUF).uri());
        if (artifacts.containsKey(ModelFormat.SAFETENSORS))
            return Path.of(artifacts.get(ModelFormat.SAFETENSORS).uri());
        if (artifacts.containsKey(ModelFormat.LITERT))
            return Path.of(artifacts.get(ModelFormat.LITERT).uri());
        return Path.of(artifacts.get(ModelFormat.PYTORCH).uri());
    }

    private String selectPrimaryFormat(Map<ModelFormat, ArtifactLocation> artifacts) {
        if (artifacts.containsKey(ModelFormat.GGUF)) return "GGUF";
        if (artifacts.containsKey(ModelFormat.SAFETENSORS)) return "SAFETENSORS";
        if (artifacts.containsKey(ModelFormat.LITERT)) return "LITERT";
        return "PYTORCH";
    }

    private boolean isAutoDownloadEnabled() {
        try {
            return org.eclipse.microprofile.config.ConfigProvider.getConfig()
                    .getOptionalValue("gollek.models.auto-download-enabled", Boolean.class)
                    .orElseGet(() -> config != null && config.autoDownload());
        } catch (Exception e) {
            return config != null && config.autoDownload();
        }
    }

    private Path findBestLocalArtifact(Path modelDir, ModelFormat format) {
        if (modelDir == null || !Files.exists(modelDir)) return null;
        try (var files = Files.list(modelDir)) {
            return files.filter(Files::isRegularFile)
                    .filter(p -> formatMatches(p.getFileName().toString(), format))
                    .sorted((a, b) -> Integer.compare(priority(b.getFileName().toString()),
                            priority(a.getFileName().toString())))
                    .findFirst().orElse(null);
        } catch (Exception e) {
            return null;
        }
    }

    private boolean isGgufFile(String name) {
        return name != null && name.toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    private boolean isSafetensorFile(String name) {
        if (name == null) return false;
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".safetensors") || lower.endsWith(".safetensor");
    }

    private boolean isPytorchFile(String name) {
        if (name == null) return false;
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".bin") || lower.endsWith(".pt") || lower.endsWith(".pth");
    }

    private boolean isLitertFile(String name) {
        if (name == null) return false;
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".litertlm") || lower.endsWith(".task");
    }

    private String fileNameOnly(String remotePath) {
        int idx = remotePath.lastIndexOf('/');
        return idx >= 0 ? remotePath.substring(idx + 1) : remotePath;
    }

    private void downloadSidecars(String slug, Path targetDir, List<String> availableFiles) {
        for (String sidecar : List.of(
                "config.json", "tokenizer.json", "tokenizer.model",
                "tokenizer_config.json", "generation_config.json",
                "special_tokens_map.json")) {
            Optional<String> remote = availableFiles.stream()
                    .filter(name -> name.equalsIgnoreCase(sidecar))
                    .findFirst();
            if (remote.isEmpty()) continue;
            Path target = targetDir.resolve(fileNameOnly(remote.get()));
            if (Files.exists(target)) continue;
            try {
                client.downloadFile(slug, remote.get(), target, progressPrinter(slug, remote.get()));
            } catch (Exception e) {
                LOG.warnf("Failed to download sidecar %s: %s", remote.get(), e.getMessage());
            }
        }
    }

    private DownloadProgressListener progressPrinter(String slug, String filename) {
        return new DownloadProgressListener() {
            private long lastUpdate = 0;
            private double lastProgress = -1;

            @Override
            public void onStart(long totalBytes) {
                LOG.infof("Downloading %s from Kaggle %s...", filename, slug);
            }

            @Override
            public void onProgress(long downloadedBytes, long totalBytes, double progress) {
                long now = System.currentTimeMillis();
                if (now - lastUpdate < 500 && Math.abs(progress - lastProgress) < 0.01) return;
                lastUpdate = now;
                lastProgress = progress;

                if (totalBytes > 0) {
                    int percent = (int) Math.min(100, Math.round(progress * 100));
                    System.out.printf("\rKaggle download %s: %d%% (%d/%d MB)   ",
                            filename, percent,
                            downloadedBytes / 1024 / 1024,
                            totalBytes / 1024 / 1024);
                } else {
                    System.out.printf("\rKaggle download %s: %d MB   ", filename, downloadedBytes / 1024 / 1024);
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
                LOG.warnf("Download failed for %s/%s: %s", slug, filename,
                        error != null ? error.getMessage() : "unknown error");
            }
        };
    }
}
