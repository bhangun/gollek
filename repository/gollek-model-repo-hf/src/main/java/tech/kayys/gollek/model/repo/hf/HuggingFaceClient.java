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

    /**
     * Get model information
     */
    public HuggingFaceModelInfo getModelInfo(String modelId) throws IOException, InterruptedException {
        String url = config.baseUrl() + "/api/models/" + modelId;

        // Fetch model info with siblings (files)
        return fetchModelInfo(url);
    }

    /**
     * List files in a model repository
     */
    public List<String> listFiles(String modelId) throws IOException, InterruptedException {
        HuggingFaceModelInfo info = getModelInfo(modelId);
        if (info.getFiles() == null) {
            return List.of();
        }
        return info.getFiles().stream()
                .map(HuggingFaceModelInfo.ModelFile::getFilename)
                .toList();
    }

    private HuggingFaceModelInfo fetchModelInfo(String url) throws IOException, InterruptedException {
        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(config.timeoutSeconds()))
                .header("User-Agent", config.userAgent())
                .GET();

        // Add auth token if configured and not blank
        config.token()
                .filter(t -> !t.isBlank())
                .ifPresent(token -> requestBuilder.header("Authorization", "Bearer " + token));

        HttpRequest request = requestBuilder.build();

        LOG.infof("Fetching model info from: %s", url);

        HttpResponse<String> response = httpClient.send(
                request,
                HttpResponse.BodyHandlers.ofString());

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

        String url = String.format(
                "%s/%s/resolve/main/%s",
                config.baseUrl(),
                modelId,
                filename);

        LOG.infof("Downloading: %s from %s", filename, modelId);

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(config.timeoutSeconds()))
                .header("User-Agent", config.userAgent())
                .GET();

        // Add auth token if configured and not blank
        config.token()
                .filter(t -> !t.isBlank())
                .ifPresent(token -> requestBuilder.header("Authorization", "Bearer " + token));

        HttpRequest request = requestBuilder.build();

        // First, get content length
        HttpResponse<InputStream> response = httpClient.send(
                request,
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() != 200) {
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

        long contentLength = response.headers()
                .firstValueAsLong("Content-Length")
                .orElse(-1L);

        // Check if file already exists and matches size
        if (Files.exists(targetPath) && contentLength > 0) {
            long existingSize = Files.size(targetPath);
            if (existingSize == contentLength) {
                LOG.infof("File already exists and matches size (%d bytes), skipping download.", existingSize);
                if (progressListener != null) {
                    // Report 100% progress immediately
                    progressListener.onProgress(contentLength, contentLength, 1.0);
                }
                return;
            } else {
                LOG.infof("File exists but size mismatch (expected %d, found %d), re-downloading.", contentLength,
                        existingSize);
            }
        }

        // Download with progress tracking
        try (InputStream is = response.body()) {
            downloadWithProgress(is, targetPath, contentLength, progressListener);
        }

        LOG.infof("Downloaded: %s (%d bytes)", filename, Files.size(targetPath));
    }

    /**
     * Download entire repository (essential files for conversion)
     */
    public void downloadRepository(String modelId, java.nio.file.Path targetDir,
            DownloadProgressListener progressListener)
            throws IOException, InterruptedException {
        List<String> files = listFiles(modelId);

        // Filter for essential files if you want to optimize,
        // but for conversion we might need everything except maybe other safetensors if
        // checking strictly?
        // For simplicity, download everything or at least:
        // - config.json, tokenizer.json, *.safetensors, *.bin, tokenizer_config.json

        for (String file : files) {
            // Skip .git attributes or large hidden files if any
            if (file.startsWith("."))
                continue;

            downloadFile(modelId, file, targetDir.resolve(file), progressListener);
        }
    }

    private void downloadWithProgress(
            InputStream inputStream,
            java.nio.file.Path targetPath,
            long totalBytes,
            DownloadProgressListener progressListener) throws IOException, InterruptedException {

        Files.createDirectories(targetPath.getParent());

        // Use a temporary file for download to avoid partial files corrupting state
        java.nio.file.Path tempPath = targetPath.resolveSibling(targetPath.getFileName() + ".part");

        byte[] buffer = new byte[8192];
        long downloadedBytes = 0;
        int bytesRead;

        // Register shutdown hook to interrupt download on Ctrl+C
        Thread currentThread = Thread.currentThread();
        Thread shutdownHook = new Thread(() -> {
            try {
                // Interrupt the download thread to stop the loop/IO
                currentThread.interrupt();
                // Force immediate exit with standard Ctrl+C exit code
                System.exit(130);
            } catch (Exception e) {
                // Best effort
            }
        });
        Runtime.getRuntime().addShutdownHook(shutdownHook);

        try (var outputStream = Files.newOutputStream(tempPath)) {
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
            // Clean up partial file on error or interruption
            try {
                Files.deleteIfExists(tempPath);
            } catch (IOException ignored) {
            }
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
}
