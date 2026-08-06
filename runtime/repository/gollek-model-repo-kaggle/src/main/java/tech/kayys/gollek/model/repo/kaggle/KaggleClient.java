package tech.kayys.gollek.model.repo.kaggle;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.model.download.DownloadProgressListener;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.List;
import java.util.Optional;

/**
 * Client for Kaggle API — lists files and downloads model artifacts.
 */
@ApplicationScoped
public class KaggleClient {

    private static final Logger LOG = Logger.getLogger(KaggleClient.class);

    @Inject
    KaggleConfig config;

    @Inject
    ObjectMapper objectMapper;

    private final HttpClient httpClient;

    public KaggleClient() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
    }

    @PostConstruct
    void init() {
        objectMapper.registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule());
    }

    /**
     * List files in a model repository.
     *
     * @param modelSlug e.g. "google/gemma/2b"
     * @return list of filenames
     */
    public List<String> listFiles(String modelSlug) throws IOException, InterruptedException {
        String url = String.format("%s/v1/models/%s/list", config.apiBaseUrl(), modelSlug);
        LOG.infof("Listing files from Kaggle model: %s", modelSlug);

        HttpResponse<String> response = httpClient.send(
                buildGetRequest(url),
                HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException(String.format(
                    "Failed to list Kaggle model files: %d — %s",
                    response.statusCode(), response.body()));
        }

        // Parse response — Kaggle returns {"files": [{"path": "...", "size": ...}]}
        KaggleFileList fileList = objectMapper.readValue(response.body(), KaggleFileList.class);
        return fileList.files().stream()
                .map(f -> f.path())
                .toList();
    }

    /**
     * Download a specific file from a model.
     */
    public void downloadFile(
            String modelSlug,
            String filename,
            Path targetPath,
            DownloadProgressListener progressListener) throws IOException, InterruptedException {

        String url = String.format(
                "%s/v1/models/%s/download/%s",
                config.apiBaseUrl(), modelSlug, filename);

        LOG.infof("Downloading: %s from Kaggle model %s", filename, modelSlug);

        HttpResponse<InputStream> response = httpClient.send(
                buildGetRequest(url),
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() != 200) {
            throw new IOException(String.format(
                    "Failed to download file: %d", response.statusCode()));
        }

        long contentLength = response.headers()
                .firstValueAsLong("Content-Length")
                .orElse(-1L);

        if (Files.exists(targetPath) && contentLength > 0) {
            long existingSize = Files.size(targetPath);
            if (existingSize == contentLength) {
                LOG.infof("File already exists and matches size (%d bytes), skipping.", existingSize);
                if (progressListener != null) {
                    progressListener.onProgress(contentLength, contentLength, 1.0);
                }
                return;
            }
        }

        try (InputStream is = response.body()) {
            downloadWithProgress(is, targetPath, contentLength, progressListener);
        }

        LOG.infof("Downloaded: %s (%d bytes)", filename, Files.size(targetPath));
    }

    private void downloadWithProgress(
            InputStream inputStream,
            Path targetPath,
            long totalBytes,
            DownloadProgressListener progressListener) throws IOException, InterruptedException {

        Files.createDirectories(targetPath.getParent());
        Path tempPath = targetPath.resolveSibling(targetPath.getFileName() + ".part");

        byte[] buffer = new byte[8192];
        long downloadedBytes = 0;
        int bytesRead;

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
            Files.deleteIfExists(tempPath);
            throw e;
        }

        Files.move(tempPath, targetPath, StandardCopyOption.REPLACE_EXISTING);

        if (progressListener != null) {
            progressListener.onComplete(downloadedBytes);
        }
    }

    private HttpRequest buildGetRequest(String url) {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(config.timeoutSeconds()))
                .header("User-Agent", config.userAgent())
                .GET();

        configuredKey().ifPresent(key -> builder.header("Authorization", "Bearer " + key));
        return builder.build();
    }

    private Optional<String> configuredKey() {
        return config.token()
                .map(String::trim)
                .filter(k -> !k.isBlank() && !k.contains("${"));
    }

    // ── Inner records for JSON parsing ──────────────────────────────────

    record KaggleFileList(List<KaggleFileEntry> files) {}
    record KaggleFileEntry(String path, long size) {}
}
