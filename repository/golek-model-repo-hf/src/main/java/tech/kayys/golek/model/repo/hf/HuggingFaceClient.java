package tech.kayys.golek.model.repository.hf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.rest.client.annotation.RegisterProvider;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;
import org.jboss.logging.Logger;
import tech.kayys.wayang.inference.repository.download.DownloadProgressListener;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.time.Duration;

/**
 * Client for HuggingFace API
 */
@ApplicationScoped
public class HuggingFaceClient {

    private static final Logger LOG = Logger.getLogger(HuggingFaceClient.class);

    @Inject
    HuggingFaceConfig config;

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

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(config.timeoutSeconds()))
                .header("User-Agent", config.userAgent())
                .GET();

        // Add auth token if configured
        config.token().ifPresent(token -> requestBuilder.header("Authorization", "Bearer " + token));

        HttpRequest request = requestBuilder.build();

        LOG.infof("Fetching model info for: %s", modelId);

        HttpResponse<String> response = httpClient.send(
                request,
                HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException(String.format(
                    "Failed to fetch model info: %d - %s",
                    response.statusCode(),
                    response.body()));
        }

        return parseModelInfo(response.body());
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

        config.token().ifPresent(token -> requestBuilder.header("Authorization", "Bearer " + token));

        HttpRequest request = requestBuilder.build();

        // First, get content length
        HttpResponse<InputStream> response = httpClient.send(
                request,
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() != 200) {
            throw new IOException(String.format(
                    "Failed to download file: %d",
                    response.statusCode()));
        }

        long contentLength = response.headers()
                .firstValueAsLong("Content-Length")
                .orElse(-1L);

        // Download with progress tracking
        try (InputStream is = response.body()) {
            downloadWithProgress(is, targetPath, contentLength, progressListener);
        }

        LOG.infof("Downloaded: %s (%d bytes)", filename, Files.size(targetPath));
    }

    private void downloadWithProgress(
            InputStream inputStream,
            java.nio.file.Path targetPath,
            long totalBytes,
            DownloadProgressListener progressListener) throws IOException {

        Files.createDirectories(targetPath.getParent());

        byte[] buffer = new byte[8192];
        long downloadedBytes = 0;
        int bytesRead;

        try (var outputStream = Files.newOutputStream(targetPath)) {
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
                downloadedBytes += bytesRead;

                if (progressListener != null && totalBytes > 0) {
                    double progress = (double) downloadedBytes / totalBytes;
                    progressListener.onProgress(downloadedBytes, totalBytes, progress);
                }
            }
        }

        if (progressListener != null) {
            progressListener.onComplete(downloadedBytes);
        }
    }

    private HuggingFaceModelInfo parseModelInfo(String json) throws IOException {
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper()
                    .readValue(json, HuggingFaceModelInfo.class);
        } catch (Exception e) {
            throw new IOException("Failed to parse model info", e);
        }
    }
}