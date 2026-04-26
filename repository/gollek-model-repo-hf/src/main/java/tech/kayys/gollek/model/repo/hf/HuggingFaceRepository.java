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

import tech.kayys.gollek.model.repo.local.ManifestStore;
import tech.kayys.gollek.model.repo.local.GollekManifest;
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

    private final ManifestStore manifestStore;
    private final HuggingFaceConfig config;
    private final HuggingFaceClient client;
    private final HuggingFaceArtifactResolver resolver;
    private final HuggingFaceDownloader downloader;
    private final Path cacheDir;

    @Inject
    public HuggingFaceRepository(HuggingFaceClient client, HuggingFaceConfig config, ManifestStore manifestStore) {
        this(resolveDefaultCacheDir(), client, config, manifestStore);
    }

    public HuggingFaceRepository(Path cacheDir, HuggingFaceClient client, ManifestStore manifestStore) {
        this(cacheDir, client, null, manifestStore);
    }

    public HuggingFaceRepository(Path cacheDir, HuggingFaceClient client, HuggingFaceConfig config, ManifestStore manifestStore) {
        this.cacheDir = cacheDir;
        this.client = client;
        this.config = config;
        this.manifestStore = manifestStore;
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
        // Determine format-specific cache directory
        Path target = resolveFormatSpecificCacheDir(descriptor);
        return downloader.download(descriptor, target);
    }

    /**
     * Resolve cache directory based on model format.
     * - Safetensors: ~/.gollek/models/safetensors/{model_id}[-revision]/
     * - GGUF: ~/.gollek/models/gguf/{model_id}[-revision].gguf
     * - PyTorch/TorchScript: ~/.gollek/models/torchscript/{model_id}[-revision]/
     * - Default: ~/.gollek/models/{format}/{model_id}[-revision]/
     */
    private Path resolveFormatSpecificCacheDir(ModelDescriptor descriptor) {
        String repoId = descriptor.id();
        String revision = descriptor.metadata().get("revision");
        String manifestName = GollekManifest.computeName(repoId, revision);
        return ManifestStore.resolveBlobDir(repoId, manifestName);
    }

    @Override
    public Uni<ModelManifest> findById(String modelId, String requestId) {
        if (!isHuggingFaceModelId(modelId)) {
            return Uni.createFrom().nullItem();
        }
        
        return Uni.createFrom().item(() -> {
            try {
                String repoId = normalizeRepoId(modelId);
                String revision = config != null ? config.revision() : DEFAULT_REVISION;
                
                // 1. Check ManifestStore first
                Optional<GollekManifest> gm = manifestStore.findByModelId(repoId, revision);
                if (gm.isPresent()) {
                    return toModelManifest(gm.get(), requestId);
                }

                // 2. Trigger auto-download if enabled
                if (isAutoDownloadEnabled() && !"community".equals(requestId)) {
                    String manifestName = GollekManifest.computeName(repoId, revision);
                    Path targetFolder = ManifestStore.resolveBlobDir(repoId, manifestName);
                    
                    Path downloaded = downloadBestArtifact(repoId, targetFolder, tech.kayys.gollek.spi.model.PullOptions.DEFAULT);
                    if (downloaded != null) {
                        ModelFormat format = detectFormat(downloaded);
                        GollekManifest manifest = new GollekManifest();
                        manifest.setId(manifestName); // Use name as ID for HF consistency
                        manifest.setModelId(repoId);
                        manifest.setName(manifestName);
                        manifest.setFormat(format.name());
                        manifest.setSource("huggingface");
                        manifest.setBlobPath(downloaded.toAbsolutePath().toString());
                        manifest.setFiles(ManifestStore.listBlobFiles(Files.isRegularFile(downloaded) ? downloaded.getParent() : downloaded));
                        manifest.setCreatedAt(java.time.Instant.now());
                        manifest.setSizeBytes(Files.isRegularFile(downloaded) ? Files.size(downloaded) : 0);
                        
                        manifestStore.save(manifest);
                        return toModelManifest(manifest, requestId);
                    }
                }
                
                return null;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }).onFailure().invoke(e -> LOG.warnf("HF auto-download failed for %s: %s", modelId, e.getMessage()))
                .onFailure().recoverWithNull();
    }

    private ModelManifest toModelManifest(GollekManifest gm, String requestId) {
         ModelFormat format = ModelFormat.fromId(gm.getFormat());
         String path = gm.getBlobPath();
         
         tech.kayys.gollek.spi.model.ArtifactLocation artifact = new tech.kayys.gollek.spi.model.ArtifactLocation(
                 path != null ? Path.of(path).toUri().toString() : "", 
                 null, gm.getSizeBytes(), "application/octet-stream");

         return ModelManifest.builder()
                 .modelId(gm.getModelId())
                 .name(gm.getName())
                 .version("1.0.0")
                 .requestId(requestId != null ? requestId : "local")
                 .path(path != null ? path : "")
                 .apiKey("community")
                 .artifacts(Map.of(format, artifact))
                 .metadata(Map.of(
                         "source", "huggingface",
                         "repo", gm.getModelId(),
                         "format", format.name(),
                         "manifestId", gm.getId()))
                 .build();
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
    public Uni<ModelManifest> pull(String modelId, tech.kayys.gollek.spi.model.PullOptions options) {
        return Uni.createFrom().item(() -> {
            try {
                String repoId = normalizeRepoId(modelId);
                String revision = options.getRevision() != null ? options.getRevision() : (config != null ? config.revision() : DEFAULT_REVISION);
                String manifestName = GollekManifest.computeName(repoId, revision);
                
                Path targetFolder = ManifestStore.resolveBlobDir(repoId, manifestName);
                
                Path downloaded = downloadBestArtifact(repoId, targetFolder, options);
                ModelFormat format = detectFormat(downloaded);
                
                GollekManifest manifest = new GollekManifest();
                manifest.setId(manifestName);
                manifest.setModelId(repoId);
                manifest.setName(manifestName);
                manifest.setFormat(format.name());
                manifest.setSource("huggingface");
                manifest.setBlobPath(downloaded.toAbsolutePath().toString());
                manifest.setFiles(ManifestStore.listBlobFiles(Files.isRegularFile(downloaded) ? downloaded.getParent() : downloaded));
                manifest.setCreatedAt(java.time.Instant.now());
                if (Files.isRegularFile(downloaded)) {
                    manifest.setSizeBytes(Files.size(downloaded));
                }

                manifestStore.save(manifest);
                return toModelManifest(manifest, "pull");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }


    @Override
    public boolean isCached(String modelId, ModelFormat format) {
        if (!isHuggingFaceModelId(modelId)) {
            return false;
        }
        String repoId = normalizeRepoId(modelId);
        
        // Check format-specific directory
        Path modelDir = switch (format) {
            case ModelFormat.GGUF -> cacheDir.resolve("gguf").resolve(repoId + ".gguf");
            case ModelFormat.SAFETENSORS -> cacheDir.resolve("safetensors").resolve(repoId);
            case ModelFormat.TORCHSCRIPT -> cacheDir.resolve("torchscript").resolve(repoId);
            case ModelFormat.LITERT -> cacheDir.resolve("litert").resolve(repoId);
            default -> cacheDir.resolve("safetensors").resolve(repoId);
        };
        
        Path artifact = findBestLocalArtifact(modelDir, format);
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
        return modelId != null && modelId.startsWith("hf:");
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

    private Path downloadBestArtifact(String repoId, Path targetDir, tech.kayys.gollek.spi.model.PullOptions options) throws Exception {
        Files.createDirectories(targetDir);
        String revision = options.getRevision() != null ? options.getRevision() : (config != null ? config.revision() : DEFAULT_REVISION);
        List<String> files = client.listFiles(repoId, revision);

        // 0. Check for pipeline model (Stable Diffusion / Transformers pipeline)
        boolean isPipeline = files.stream().anyMatch(name -> name.endsWith("model_index.json"));
        if (isPipeline) {
            LOG.infof("Pipeline model detected for %s. Downloading all mandatory components...", repoId);
            Path primary = null;
            
            // Smart Filter: Deduplicate formats within directories (prefer .safetensors over .bin)
            List<String> toDownload = filterSmartWeights(files);
            
            int downloadedCount = 0;
            for (String file : toDownload) {
                if (isCheckpointRelevant(file)) {
                    Path target = targetDir.resolve(file);
                    if (target.getParent() != null) {
                        Files.createDirectories(target.getParent());
                    }
                    if (options.isForce() || !Files.exists(target)) {
                        try {
                            client.downloadFile(repoId, file, options.getRevision(), target, progressPrinter(repoId, file));
                            downloadedCount++;
                        } catch (Exception e) {
                            LOG.warnf("Failed to download optional component [%s]: %s", file, e.getMessage());
                        }
                    }
                    if (primary == null && (isSafetensorFile(file) || isGgufFile(file))) {
                        primary = target;
                    }
                }
            }
            LOG.infof("Pipeline sync complete for %s. %d files downloaded/verified.", repoId, downloadedCount);
            if (primary != null) {
                validateDownloadedArtifact(primary);
                return primary;
            }
            // If No obvious primary found but it's a pipeline, return the model_index as anchor
            Path index = targetDir.resolve("model_index.json");
            if (Files.exists(index)) return index;
        }

        // 1. Try GGUF first if we are in a GGUF-preferred context
        Optional<String> ggufFile = files.stream()
                .filter(this::isGgufFile)
                .findFirst();

        if (ggufFile.isPresent()) {
            Path target = targetDir.resolve(ggufFile.get());
            if (target.getParent() != null) Files.createDirectories(target.getParent());
            if (!Files.exists(target)) {
                client.downloadFile(repoId, ggufFile.get(), revision, target, progressPrinter(repoId, ggufFile.get()));
            }
            downloadSidecars(repoId, targetDir, files, revision);
            validateDownloadedArtifact(target);
            return target;
        }

        // 2. Fallback to safetensors - prefer model.safetensors
        Optional<String> exact = files.stream()
                .filter(name -> "model.safetensors".equalsIgnoreCase(name) || "model.safetensor".equalsIgnoreCase(name))
                .findFirst();

        if (exact.isPresent()) {
            Path target = targetDir.resolve(exact.get());
            if (target.getParent() != null) Files.createDirectories(target.getParent());
            if (!Files.exists(target)) {
                client.downloadFile(repoId, exact.get(), revision, target, progressPrinter(repoId, exact.get()));
            }
            downloadSidecars(repoId, targetDir, files, revision);
            validateDownloadedArtifact(target);
            return target;
        }

        // 3. Try any .safetensors file
        Optional<String> genericSafetensors = files.stream()
                .filter(this::isSafetensorFile)
                .findFirst();

        if (genericSafetensors.isPresent()) {
            Path target = targetDir.resolve(genericSafetensors.get());
            if (target.getParent() != null) Files.createDirectories(target.getParent());
            if (!Files.exists(target)) {
                client.downloadFile(repoId, genericSafetensors.get(), revision, target,
                        progressPrinter(repoId, genericSafetensors.get()));
            }
            downloadSidecars(repoId, targetDir, files, revision);
            validateDownloadedArtifact(target);
            return target;
        }

        // 4. Fallback to PyTorch bin files
        Optional<String> pytorchFile = files.stream()
                .filter(name -> name.toLowerCase().endsWith(".bin") || name.toLowerCase().endsWith(".pt") || name.toLowerCase().endsWith(".pth"))
                .findFirst();

        if (pytorchFile.isPresent()) {
            Path target = targetDir.resolve(pytorchFile.get());
            if (target.getParent() != null) Files.createDirectories(target.getParent());
            if (!Files.exists(target)) {
                client.downloadFile(repoId, pytorchFile.get(), revision, target,
                        progressPrinter(repoId, pytorchFile.get()));
            }
            downloadSidecars(repoId, targetDir, files, revision);
            validateDownloadedArtifact(target);
            return target;
        }

        // 5. Try LiteRT files
        Optional<String> litertFile = files.stream()
                .filter(this::isLitertFile)
                .findFirst();

        if (litertFile.isPresent()) {
            Path target = targetDir.resolve(litertFile.get());
            if (target.getParent() != null) Files.createDirectories(target.getParent());
            if (!Files.exists(target)) {
                client.downloadFile(repoId, litertFile.get(), revision, target,
                        progressPrinter(repoId, litertFile.get()));
            }
            downloadSidecars(repoId, targetDir, files, revision);
            validateDownloadedArtifact(target);
            return target;
        }

        return null;
    }

    private boolean isAutoDownloadEnabled() {
        // Check global flag first, then fallback to HF specific config
        try {
            return org.eclipse.microprofile.config.ConfigProvider.getConfig()
                    .getOptionalValue("gollek.models.auto-download-enabled", Boolean.class)
                    .orElseGet(() -> config != null && config.autoDownload());
        } catch (Exception e) {
            return config != null && config.autoDownload();
        }
    }

    private ArtifactLocation toArtifactLocation(Path path) throws java.io.IOException {
        String uri = path.toUri().toString();
        long size = Files.size(path);
        return new ArtifactLocation(uri, null, size, "application/octet-stream");
    }

    private Path findBestLocalArtifact(Path modelDir, ModelFormat format) {
        if (modelDir == null || !Files.exists(modelDir)) {
            return null;
        }
        try (var files = Files.list(modelDir)) {
            return files
                    .filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString();
                        if (format == ModelFormat.GGUF)
                            return isGgufFile(name);
                        if (format == ModelFormat.SAFETENSORS)
                            return isSafetensorFile(name);
                        if (format == ModelFormat.LITERT)
                            return isLitertFile(name);
                        return false;
                    })
                    .sorted((a, b) -> Integer.compare(priority(b.getFileName().toString()),
                            priority(a.getFileName().toString())))
                    .findFirst()
                    .orElse(null);
        } catch (Exception e) {
            return null;
        }
    }

    private boolean isGgufFile(String name) {
        return name != null && name.toLowerCase(Locale.ROOT).endsWith(".gguf");
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

    private boolean isOnnxFile(String name) {
        if (name == null) {
            return false;
        }
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".onnx");
    }

    private boolean isLitertFile(String name) {
        if (name == null) {
            return false;
        }
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".litertlm") || lower.endsWith(".task");
    }

    private int priority(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        if (lower.equals("model.safetensors") || lower.equals("model.safetensor") || lower.endsWith("-web.task")) {
            return 100;
        }
        if (lower.endsWith(".safetensors") || lower.endsWith(".safetensor") || lower.endsWith(".litertlm") || lower.endsWith(".task")) {
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
        if (name.endsWith(".gguf")) {
            return ModelFormat.GGUF;
        }
        if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
            return ModelFormat.SAFETENSORS;
        }
        if (name.endsWith(".onnx")) {
            return ModelFormat.ONNX;
        }
        if (name.endsWith(".litertlm") || name.endsWith(".task")) {
            return ModelFormat.LITERT;
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
            private long lastUpdate = 0;
            private double lastProgress = -1;

            @Override
            public void onStart(long totalBytes) {
                LOG.infof("Downloading %s from %s...", filename, repoId);
            }

            @Override
            public void onProgress(long downloadedBytes, long totalBytes, double progress) {
                long now = System.currentTimeMillis();
                // Update at most once every 500ms or if progress changed significantly (>1%)
                if (now - lastUpdate < 500 && Math.abs(progress - lastProgress) < 0.01) {
                    return;
                }
                lastUpdate = now;
                lastProgress = progress;

                if (totalBytes > 0) {
                    int percent = (int) Math.min(100, Math.round(progress * 100));
                    System.out.printf("\rHF download %s: %d%% (%d/%d MB)   ",
                            filename,
                            percent,
                            downloadedBytes / 1024 / 1024,
                            totalBytes / 1024 / 1024);
                } else {
                    System.out.printf("\rHF download %s: %d MB   ", filename, downloadedBytes / 1024 / 1024);
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

    private void downloadSidecars(String repoId, Path targetDir, List<String> availableFiles, String revision) {
        for (String sidecar : sidecarCandidates()) {
            Optional<String> remote = availableFiles.stream()
                    .filter(name -> name.equalsIgnoreCase(sidecar))
                    .findFirst();
            if (remote.isEmpty()) {
                continue;
            }
            Path target = targetDir.resolve(remote.get());
            if (target.getParent() != null) {
                try {
                    Files.createDirectories(target.getParent());
                } catch (java.io.IOException e) {
                    LOG.warnf("Failed to create sidecar directory %s: %s", target.getParent(), e.getMessage());
                }
            }
            if (Files.exists(target)) {
                continue;
            }
            try {
                client.downloadFile(repoId, remote.get(), revision, target, progressPrinter(repoId, remote.get()));
            } catch (Exception e) {
                LOG.warnf("Failed to download sidecar %s for %s: %s", remote.get(), repoId, e.getMessage());
            }
        }
    }

    private boolean isCheckpointRelevant(String name) {
        if (name == null || name.startsWith(".")) return false;
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".safetensors") || lower.endsWith(".bin") || lower.endsWith(".pt") || lower.endsWith(".pth")
                || lower.endsWith(".json") || lower.endsWith(".model") || lower.endsWith(".txt") || lower.endsWith(".spm")
                || lower.endsWith(".msgpack") || lower.endsWith(".gguf") || lower.endsWith(".ckpt") || lower.endsWith(".onnx");
    }

    private List<String> filterSmartWeights(List<String> files) {
        Map<String, List<String>> dirToFiles = new java.util.HashMap<>();
        for (String f : files) {
            int lastSlash = f.lastIndexOf('/');
            String dir = lastSlash >= 0 ? f.substring(0, lastSlash) : "";
            dirToFiles.computeIfAbsent(dir, k -> new java.util.ArrayList<>()).add(f);
        }

        List<String> result = new java.util.ArrayList<>();
        for (Map.Entry<String, List<String>> entry : dirToFiles.entrySet()) {
            List<String> dirFiles = entry.getValue();
            boolean hasSafetensors = dirFiles.stream().anyMatch(f -> f.endsWith(".safetensors"));
            
            for (String f : dirFiles) {
                if (hasSafetensors && (f.endsWith(".bin") || f.endsWith(".pt") || f.endsWith(".pth"))) {
                    // Skip redundant formats if safetensors exists in the same folder
                    continue;
                }
                result.add(f);
            }
        }
        return result;
    }

    private void ensureLocalSidecars(String repoId, Path targetDir) {
        try {
            String revision = config != null ? config.revision() : DEFAULT_REVISION;
            List<String> files = client.listFiles(repoId, revision);
            downloadSidecars(repoId, targetDir, files, revision);
        } catch (Exception e) {
            LOG.debugf("Unable to refresh sidecars for %s: %s", repoId, e.getMessage());
        }
    }

    private List<String> sidecarCandidates() {
        return List.of(
                "config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer.spm",
                "tokenizer_config.json",
                "generation_config.json",
                "special_tokens_map.json",
                "processor_config.json",
                "preprocessor_config.json");
    }
}
