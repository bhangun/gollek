package tech.kayys.golek.provider.huggingface;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.model.download.DownloadProgressListener;

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
import java.util.Map;
import java.util.Optional;

/**
 * Client for HuggingFace Hub API
 */
@ApplicationScoped
public class HuggingFaceClient {

    private static final Logger LOG = Logger.getLogger(HuggingFaceClient.class);
    private static final String API_BASE = "https://huggingface.co/api/models/";
    private static final String RESOLVE_BASE = "https://huggingface.co/";

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    @Inject
    public HuggingFaceClient(HttpClient httpClient) {
        this.httpClient = httpClient;
        this.objectMapper = new ObjectMapper();
    }

    public HuggingFaceClient() {
        this(HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build());
    }

    /**
     * Get model metadata from HF Hub
     */
    public Map<String, Object> getModelMetadata(String modelId, Optional<String> token)
            throws IOException, InterruptedException {
        String url = API_BASE + modelId;

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .GET();

        token.ifPresent(t -> requestBuilder.header("Authorization", "Bearer " + t));

        HttpRequest request = requestBuilder.build();
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException("Failed to fetch model metadata: " + response.statusCode() + " " + response.body());
        }

        return objectMapper.readValue(response.body(), Map.class);
    }

    /**
     * Download file from HF Hub
     */
    public void downloadFile(String modelId, String filename, Path targetPath, Optional<String> token,
            DownloadProgressListener listener) throws IOException, InterruptedException {
        // HF Resolve URL: https://huggingface.co/{modelId}/resolve/main/{filename}
        String url = RESOLVE_BASE + modelId + "/resolve/main/" + filename;

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .GET();

        token.ifPresent(t -> requestBuilder.header("Authorization", "Bearer " + t));

        HttpRequest request = requestBuilder.build();

        LOG.infof("Downloading %s from %s to %s", filename, modelId, targetPath);

        HttpResponse<InputStream> response = httpClient.send(request, HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() != 200) {
            throw new IOException("Failed to download file: " + response.statusCode());
        }

        long totalBytes = response.headers().firstValueAsLong("Content-Length").orElse(-1L);

        if (listener != null) {
            listener.onStart(totalBytes);
        }

        Files.createDirectories(targetPath.getParent());
        Path tempPath = targetPath.resolveSibling(targetPath.getFileName() + ".tmp");

        try (InputStream is = response.body();
                var os = Files.newOutputStream(tempPath)) {

            byte[] buffer = new byte[8192];
            long downloadedBytes = 0;
            int bytesRead;

            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
                downloadedBytes += bytesRead;

                if (listener != null && totalBytes > 0) {
                    double progress = (double) downloadedBytes / totalBytes;
                    listener.onProgress(downloadedBytes, totalBytes, progress);
                }
            }
        }

        Files.move(tempPath, targetPath, StandardCopyOption.REPLACE_EXISTING);

        if (listener != null) {
            listener.onComplete(totalBytes);
        }

        LOG.infof("Download complete: %s", targetPath);
    }

    /**
     * Download a specific byte range of a file from HF Hub
     */
    public InputStream downloadRange(String modelId, String filename, long startByte, long endByte,
            Optional<String> token) throws IOException, InterruptedException {
        String url = RESOLVE_BASE + modelId + "/resolve/main/" + filename;

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Range", "bytes=" + startByte + "-" + endByte)
                .GET();

        token.ifPresent(t -> requestBuilder.header("Authorization", "Bearer " + t));

        HttpRequest request = requestBuilder.build();

        // Execute with retry
        return executeWithRetry(() -> {
            try {
                HttpResponse<InputStream> response = httpClient.send(request,
                        HttpResponse.BodyHandlers.ofInputStream());
                if (response.statusCode() != 206 && response.statusCode() != 200) {
                    throw new IOException("Failed to download range: " + response.statusCode());
                }
                return response.body();
            } catch (IOException | InterruptedException e) {
                throw new RuntimeException(e);
            }
        });
    }

    private <T> T executeWithRetry(java.util.function.Supplier<T> action) {
        int maxRetries = 3;
        long backoff = 1000;

        for (int i = 0; i < maxRetries; i++) {
            try {
                return action.get();
            } catch (Exception e) {
                if (i == maxRetries - 1)
                    throw e;
                LOG.warnf("Action failed (attempt %d/%d), retrying in %dms: %s", i + 1, maxRetries, backoff,
                        e.getMessage());
                try {
                    Thread.sleep(backoff);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(ie);
                }
                backoff *= 2;
            }
        }
        return null; // Should not reach here
    }
}
