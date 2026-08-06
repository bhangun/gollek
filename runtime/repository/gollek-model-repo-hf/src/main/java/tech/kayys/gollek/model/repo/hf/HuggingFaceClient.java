package tech.kayys.gollek.model.repo.hf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.model.download.DownloadProgressListener;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.List;

/**
 * Client for HuggingFace API
 */
@ApplicationScoped
public class HuggingFaceClient {

    private static final Logger LOG = Logger.getLogger(HuggingFaceClient.class);

    @Inject
    HuggingFaceConfig config;

    @Inject
    ObjectMapper objectMapper;

    private final HttpClient httpClient;

    public HuggingFaceClient() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
    }

    @jakarta.annotation.PostConstruct
    void init() {
        objectMapper.registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule());
    }

    /**
     * Get model information
     */
    public HuggingFaceModelInfo getModelInfo(String modelId) throws IOException, InterruptedException {
        return getModelInfo(modelId, config.revision());
    }

    public HuggingFaceModelInfo getModelInfo(String modelId, String revision) throws IOException, InterruptedException {
        String effectiveRevision = revision != null ? revision : config.revision();
        String url;
        if (effectiveRevision != null && !effectiveRevision.trim().isEmpty() && !effectiveRevision.equalsIgnoreCase("main") && !effectiveRevision.equalsIgnoreCase("master")) {
            url = config.baseUrl() + "/api/models/" + modelId + "/revision/" + java.net.URLEncoder.encode(effectiveRevision.trim(), java.nio.charset.StandardCharsets.UTF_8) + "?siblings=true";
        } else {
            url = config.baseUrl() + "/api/models/" + modelId + "?siblings=true";
        }

        // Fetch model info with siblings (files)
        return fetchModelInfo(url);
    }

    /**
     * List files in a model repository
     */
    public List<String> listFiles(String modelId) throws IOException, InterruptedException {
        return listFiles(modelId, config.revision());
    }

    public List<String> listFiles(String modelId, String revision) throws IOException, InterruptedException {
        HuggingFaceModelInfo info = getModelInfo(modelId, revision);
        if (info.getFiles() == null) {
            return List.of();
        }
        return info.getFiles().stream()
                .map(HuggingFaceModelInfo.ModelFile::getFilename)
                .toList();
    }

    private HuggingFaceModelInfo fetchModelInfo(String url) throws IOException, InterruptedException {
        LOG.infof("Fetching model info from: %s", url);

        HttpResponse<String> response = httpClient.send(
                buildGetRequest(url, true),
                HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 401 && hasUsableToken()) {
            response = httpClient.send(
                    buildGetRequest(url, true),
                    HttpResponse.BodyHandlers.ofString());
        }

        if (response.statusCode() != 200) {
            String body = response.body();
            if (body == null) {
                body = "";
            }
            body = body.replaceAll("\\s+", " ").trim();
            if (body.length() > 512) {
                body = body.substring(0, 512) + "...";
            }
            throw new IOException(String.format(
                    "Failed to fetch model info: %d - %s",
                    response.statusCode(),
                    body));
        }

        String json = response.body();
        try {
            return parseModelInfo(json);
        } catch (IOException e) {
            LOG.errorf("Failed to parse HuggingFace model info. Body: %s", json);
            throw e;
        }
    }

    /**
     * Download a specific file from a model
     */
    public void downloadFile(
            String modelId,
            String filename,
            java.nio.file.Path targetPath,
            DownloadProgressListener progressListener) throws IOException, InterruptedException {
        downloadFile(modelId, filename, config.revision(), targetPath, progressListener);
    }

    public void downloadFile(
            String modelId,
            String filename,
            String revision,
            java.nio.file.Path targetPath,
            DownloadProgressListener progressListener) throws IOException, InterruptedException {
        String effectiveRevision = revision != null ? revision : config.revision();
        String url = String.format(
                "%s/%s/resolve/%s/%s",
                config.baseUrl(),
                modelId,
                effectiveRevision,
                filename);

        LOG.infof("Downloading: %s from %s", filename, modelId);

        // First, check total size using a HEAD request to verify existing files without keeping connections open
        HttpRequest headRequest = buildRequestBuilder(url, true, 600)
                .method("HEAD", HttpRequest.BodyPublishers.noBody())
                .build();
        HttpResponse<Void> headResponse = httpClient.send(headRequest, HttpResponse.BodyHandlers.discarding());

        if (headResponse.statusCode() == 401 && hasUsableToken()) {
            headResponse = httpClient.send(headRequest, HttpResponse.BodyHandlers.discarding());
        }

        long totalSize = headResponse.headers().firstValueAsLong("Content-Length").orElse(-1L);

        // Check if fully downloaded file already exists and matches size
        if (Files.exists(targetPath) && totalSize > 0) {
            long existingSize = Files.size(targetPath);
            if (existingSize == totalSize) {
                LOG.infof("File already exists and matches size (%d bytes), skipping download.", existingSize);
                if (progressListener != null) {
                    // Report 100% progress immediately
                    progressListener.onProgress(totalSize, totalSize, 1.0);
                }
                return;
            } else {
                LOG.infof("File exists but size mismatch (expected %d, found %d), re-downloading.", totalSize,
                        existingSize);
            }
        }

        // Check for partial download
        java.nio.file.Path tempPath = targetPath.resolveSibling(targetPath.getFileName() + ".part");
        long existingPartSize = Files.exists(tempPath) ? Files.size(tempPath) : 0;

        HttpRequest.Builder getBuilder = buildRequestBuilder(url, true, 600).GET();
        if (existingPartSize > 0 && totalSize > 0 && existingPartSize < totalSize) {
            getBuilder.header("Range", "bytes=" + existingPartSize + "-");
        } else if (existingPartSize >= totalSize && totalSize > 0) {
            // Part file is larger than total size (maybe corrupted or changed on server)
            existingPartSize = 0;
            Files.deleteIfExists(tempPath);
        }

        HttpResponse<InputStream> response = httpClient.send(
                getBuilder.build(),
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() == 401 && hasUsableToken()) {
            try (InputStream ignored = response.body()) {
                // close first response before retry
            } catch (Exception ignored) {
            }
            response = httpClient.send(
                    getBuilder.build(),
                    HttpResponse.BodyHandlers.ofInputStream());
        }

        if (response.statusCode() != 200 && response.statusCode() != 206) {
            String detail = "";
            try (InputStream errorBody = response.body()) {
                byte[] bytes = errorBody.readNBytes(512);
                if (bytes.length > 0) {
                    detail = " - " + new String(bytes, java.nio.charset.StandardCharsets.UTF_8).replaceAll("\\s+", " ").trim();
                }
            } catch (Exception ignored) {
                // best effort
            }
            throw new IOException(String.format(
                    "Failed to download file: %d%s",
                    response.statusCode(),
                    detail));
        }

        boolean append = (response.statusCode() == 206);
        if (!append) {
            existingPartSize = 0;
        }
        long resolvedTotalSize = totalSize > 0 ? totalSize : response.headers().firstValueAsLong("Content-Length").orElse(-1L);

        // Download with progress tracking
        try (InputStream is = response.body()) {
            downloadWithProgress(is, targetPath, tempPath, existingPartSize, resolvedTotalSize, append, progressListener);
        }

        LOG.infof("Downloaded: %s (%d bytes)", filename, Files.size(targetPath));
    }

    /**
     * Download entire repository (essential files for conversion)
     */
    public void downloadRepository(String modelId, java.nio.file.Path targetDir,
            DownloadProgressListener progressListener)
            throws IOException, InterruptedException {
        downloadRepository(modelId, config.revision(), targetDir, progressListener);
    }

    public void downloadRepository(String modelId, String revision, java.nio.file.Path targetDir,
            DownloadProgressListener progressListener)
            throws IOException, InterruptedException {
        List<String> files = listFiles(modelId, revision);

        // Filter for essential files if you want to optimize,
        // but for conversion we might need everything except maybe other safetensors if
        // checking strictly?
        // For simplicity, download everything or at least:
        // - config.json, tokenizer.json, *.safetensors, *.bin, tokenizer_config.json

        for (String file : files) {
            // Skip .git attributes or large hidden files if any
            if (file.startsWith("."))
                continue;

            downloadFile(modelId, file, revision, targetDir.resolve(file), progressListener);
        }
    }

    private void downloadWithProgress(
            InputStream inputStream,
            java.nio.file.Path targetPath,
            java.nio.file.Path tempPath,
            long existingPartSize,
            long totalBytes,
            boolean append,
            DownloadProgressListener progressListener) throws IOException, InterruptedException {

        Files.createDirectories(targetPath.getParent());

        byte[] buffer = new byte[8192];
        long downloadedBytes = existingPartSize;
        int bytesRead;

        // Register shutdown hook to interrupt download on Ctrl+C
        Thread currentThread = Thread.currentThread();
        Thread shutdownHook = new Thread(() -> {
            try {
                // Interrupt the download thread to stop the loop/IO
                currentThread.interrupt();
            } catch (Exception e) {
                // Best effort
            }
        });
        Runtime.getRuntime().addShutdownHook(shutdownHook);

        java.nio.file.OpenOption[] options = append ?
                new java.nio.file.OpenOption[]{java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND} :
                new java.nio.file.OpenOption[]{java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.TRUNCATE_EXISTING, java.nio.file.StandardOpenOption.WRITE};

        try (var outputStream = Files.newOutputStream(tempPath, options)) {
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                if (Thread.currentThread().isInterrupted()) {
                    throw new InterruptedException("Download interrupted");
                }
                outputStream.write(buffer, 0, bytesRead);
                downloadedBytes += bytesRead;

                if (progressListener != null && totalBytes > 0) {
                    double progress = (double) downloadedBytes / totalBytes;
                    progressListener.onProgress(downloadedBytes, totalBytes, progress);
                }
            }
        } catch (IOException | InterruptedException e) {
            // We do NOT delete the partial file here anymore so it can be resumed later.
            throw e;
        } finally {
            // Remove hook if finished normally
            try {
                Runtime.getRuntime().removeShutdownHook(shutdownHook);
            } catch (IllegalStateException e) {
                // Ignore if shutdown is in progress
            }
        }

        // Move temp file to target path
        Files.move(tempPath, targetPath, StandardCopyOption.REPLACE_EXISTING);

        if (progressListener != null) {
            progressListener.onComplete(downloadedBytes);
        }
    }

    private HuggingFaceModelInfo parseModelInfo(String json) throws IOException {
        try {
            return objectMapper.readValue(json, HuggingFaceModelInfo.class);
        } catch (Exception e) {
            throw new IOException("Failed to parse model info: " + e.getMessage(), e);
        }
    }

    private HttpRequest buildGetRequest(String url, boolean withAuth) {
        return buildGetRequest(url, withAuth, config.timeoutSeconds());
    }

    private HttpRequest buildGetRequest(String url, boolean withAuth, int timeoutSeconds) {
        return buildRequestBuilder(url, withAuth, timeoutSeconds).GET().build();
    }

    private HttpRequest.Builder buildRequestBuilder(String url, boolean withAuth, int timeoutSeconds) {
        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(timeoutSeconds))
                .header("User-Agent", config.userAgent());
        if (withAuth) {
            configuredToken().ifPresent(token -> requestBuilder.header("Authorization", "Bearer " + token));
        }
        return requestBuilder;
    }

    private java.util.Optional<String> configuredToken() {
        java.util.Optional<String> configured = config.token()
                .map(String::trim)
                .filter(this::isUsableToken);
        if (configured.isPresent()) {
            return configured;
        }

        // CLI usually maps env vars into config/system props, but keep a direct
        // fallback for packaged/runtime variants where that mapping is skipped.
        String[] fallbackKeys = new String[] {
                "wayang.inference.repository.huggingface.token",
                "WAYANG_INFERENCE_REPOSITORY_HUGGINGFACE_TOKEN",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN"
        };
        for (String key : fallbackKeys) {
            String value = key.contains(".") ? System.getProperty(key) : System.getenv(key);
            if (isUsableToken(value)) {
                return java.util.Optional.of(value.trim());
            }
        }
        return java.util.Optional.empty();
    }

    private boolean hasUsableToken() {
        return configuredToken().isPresent();
    }

    private boolean isUsableToken(String token) {
        if (token == null || token.isBlank()) {
            return false;
        }
        String normalized = token.trim().toLowerCase();
        if (normalized.equals("dummy")
                || normalized.equals("null")
                || normalized.equals("none")
                || normalized.equals("changeme")
                || normalized.equals("your_token_here")) {
            return false;
        }
        return !token.contains("${");
    }
}
