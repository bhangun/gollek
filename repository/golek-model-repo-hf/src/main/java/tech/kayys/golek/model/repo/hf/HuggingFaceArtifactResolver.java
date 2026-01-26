package tech.kayys.golek.model.repository.hf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.wayang.inference.providers.loader.ArtifactMetadata;
import tech.kayys.wayang.inference.providers.loader.ArtifactResolutionException;
import tech.kayys.wayang.inference.providers.loader.ArtifactResolver;
import tech.kayys.wayang.inference.repository.download.DownloadManager;
import tech.kayys.wayang.inference.repository.download.DownloadProgressListener;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Artifact resolver for HuggingFace models.
 * 
 * Supports URIs like:
 * - hf://meta-llama/Llama-2-7b-hf
 * - hf://meta-llama/Llama-2-7b-hf/pytorch_model.bin
 * - huggingface://mistralai/Mistral-7B-v0.1
 */
@ApplicationScoped
public class HuggingFaceArtifactResolver implements ArtifactResolver {

    private static final Logger LOG = Logger.getLogger(HuggingFaceArtifactResolver.class);

    private static final Pattern HF_URI_PATTERN = Pattern.compile(
            "^(hf|huggingface)://([^/]+/[^/]+)(?:/(.+))?$");

    @Inject
    HuggingFaceClient client;

    @Inject
    DownloadManager downloadManager;

    @Inject
    HuggingFaceConfig config;

    @Override
    public boolean supports(String artifactUri) {
        return HF_URI_PATTERN.matcher(artifactUri).matches();
    }

    @Override
    public Path resolve(String artifactUri, Path targetDir) throws ArtifactResolutionException {
        try {
            return resolveAsync(artifactUri, targetDir).toCompletableFuture().join();
        } catch (Exception e) {
            throw new ArtifactResolutionException(
                    artifactUri,
                    "Failed to resolve HuggingFace artifact: " + e.getMessage(),
                    ArtifactResolutionException.ErrorType.NETWORK_ERROR,
                    e);
        }
    }

    @Override
    public CompletionStage<Path> resolveAsync(String artifactUri, Path targetDir) {
        return resolveAsync(artifactUri, targetDir, null);
    }

    @Override
    public CompletionStage<Path> resolveAsync(
            String artifactUri,
            Path targetDir,
            DownloadProgressListener listener) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return doResolve(artifactUri, targetDir, listener);
            } catch (Exception e) {
                throw new RuntimeException("HuggingFace resolution failed", e);
            }
        });
    }

    private Path doResolve(
            String artifactUri,
            Path targetDir,
            DownloadProgressListener listener) throws Exception {

        Matcher matcher = HF_URI_PATTERN.matcher(artifactUri);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Invalid HuggingFace URI: " + artifactUri);
        }

        String modelId = matcher.group(2);
        String specificFile = matcher.group(3);

        LOG.infof("Resolving HuggingFace model: %s%s",
                modelId,
                specificFile != null ? " (file: " + specificFile + ")" : "");

        // Get model info
        HuggingFaceModelInfo modelInfo = client.getModelInfo(modelId);

        // Create model directory
        Path modelDir = targetDir.resolve(sanitizeModelId(modelId));
        Files.createDirectories(modelDir);

        if (specificFile != null) {
            // Download specific file
            Path filePath = modelDir.resolve(specificFile);
            if (!Files.exists(filePath)) {
                client.downloadFile(modelId, specificFile, filePath, listener);
            } else {
                LOG.infof("File already exists: %s", filePath);
            }
            return filePath;
        } else {
            // Download all essential model files
            return downloadModelFiles(modelId, modelInfo, modelDir, listener);
        }
    }

    private Path downloadModelFiles(
            String modelId,
            HuggingFaceModelInfo modelInfo,
            Path modelDir,
            DownloadProgressListener listener) throws Exception {

        // Determine which files to download based on model type
        var files = modelInfo.getFiles();
        if (files == null || files.isEmpty()) {
            throw new IllegalStateException("No files found for model: " + modelId);
        }

        // Download essential files
        for (var file : files) {
            String filename = file.getFilename();

            // Skip unnecessary files
            if (shouldSkipFile(filename)) {
                continue;
            }

            Path filePath = modelDir.resolve(filename);
            if (!Files.exists(filePath)) {
                LOG.infof("Downloading: %s", filename);
                client.downloadFile(modelId, filename, filePath, listener);
            } else {
                LOG.debugf("Skipping existing file: %s", filename);
            }
        }

        return modelDir;
    }

    private boolean shouldSkipFile(String filename) {
        // Skip certain file types
        return filename.endsWith(".md") ||
                filename.endsWith(".txt") ||
                filename.endsWith(".gitattributes") ||
                filename.startsWith(".git/");
    }

    @Override
    public boolean exists(String artifactUri) {
        try {
            Matcher matcher = HF_URI_PATTERN.matcher(artifactUri);
            if (!matcher.matches()) {
                return false;
            }

            String modelId = matcher.group(2);
            HuggingFaceModelInfo info = client.getModelInfo(modelId);
            return info != null;

        } catch (Exception e) {
            LOG.debugf("Model not found: %s", artifactUri);
            return false;
        }
    }

    @Override
    public Optional<ArtifactMetadata> getMetadata(String artifactUri) {
        try {
            Matcher matcher = HF_URI_PATTERN.matcher(artifactUri);
            if (!matcher.matches()) {
                return Optional.empty();
            }

            String modelId = matcher.group(2);
            HuggingFaceModelInfo info = client.getModelInfo(modelId);

            Map<String, String> attributes = new HashMap<>();
            attributes.put("modelId", info.getModelId());
            attributes.put("downloads", String.valueOf(info.getDownloads()));
            attributes.put("likes", String.valueOf(info.getLikes()));
            if (info.getPipelineTag() != null) {
                attributes.put("pipelineTag", info.getPipelineTag());
            }
            if (info.getLibraryName() != null) {
                attributes.put("library", info.getLibraryName());
            }

            // Calculate total size
            long totalSize = info.getFiles() != null
                    ? info.getFiles().stream()
                            .filter(f -> !shouldSkipFile(f.getFilename()))
                            .mapToLong(f -> f.getSize() != null ? f.getSize() : 0)
                            .sum()
                    : 0;

            return Optional.of(new ArtifactMetadata(
                    artifactUri,
                    info.getModelId(),
                    totalSize,
                    "model/huggingface",
                    info.getCommitSha() != null ? "sha:" + info.getCommitSha() : null,
                    info.getLastModified() != null ? info.getLastModified() : Instant.now(),
                    attributes));

        } catch (Exception e) {
            LOG.warnf("Failed to get metadata for %s: %s", artifactUri, e.getMessage());
            return Optional.empty();
        }
    }

    @Override
    public int getPriority() {
        return 100; // High priority for HF models
    }

    private String sanitizeModelId(String modelId) {
        return modelId.replace("/", "_");
    }
}