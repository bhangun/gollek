package tech.kayys.gollek.model.repo.kaggle;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.spi.download.DownloadProgressListener;

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
     * List files with metadata in a model repository.
     *
     * @param modelSlug e.g. "google/gemma/2b" or "google/gemma-2/pyTorch/gemma-2-2b-it"
     * @return list of KaggleFileEntry
     */
    public List<KaggleFileEntry> listModelFiles(String modelSlug) throws IOException, InterruptedException {
        String cleanSlug = cleanSlug(modelSlug);
        String url = String.format("%s/v1/models/%s/list", config.apiBaseUrl(), cleanSlug);
        LOG.infof("Listing files from Kaggle model: %s", cleanSlug);

        HttpResponse<String> response = httpClient.send(
                buildGetRequest(url),
                HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 401 || response.statusCode() == 403) {
            throw new IOException(String.format(
                    "Kaggle authentication failed (HTTP %d). "
                            + "Please configure your Kaggle credentials via ~/.kaggle/kaggle.json "
                            + "or KAGGLE_USERNAME and KAGGLE_KEY environment variables.",
                    response.statusCode()));
        }

        if (response.statusCode() == 404) {
            throw new IOException(String.format(
                    "Kaggle model not found: '%s' (HTTP 404). "
                            + "Please check the slug format (e.g., owner/model/framework/variation) "
                            + "and ensure it is a published Model on kaggle.com/models.",
                    cleanSlug));
        }

        if (response.statusCode() != 200) {
            throw new IOException(String.format(
                    "Failed to list Kaggle model files: %d — %s",
                    response.statusCode(), response.body()));
        }

        KaggleFileList fileList = objectMapper.readValue(response.body(), KaggleFileList.class);
        return fileList.files() != null ? fileList.files() : List.of();
    }

    /**
     * List file names in a model repository.
     *
     * @param modelSlug e.g. "google/gemma/2b"
     * @return list of filenames
     */
    public List<String> listFiles(String modelSlug) throws IOException, InterruptedException {
        return listModelFiles(modelSlug).stream()
                .map(KaggleFileEntry::path)
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

        String cleanSlug = cleanSlug(modelSlug);
        String url = String.format(
                "%s/v1/models/%s/download/%s",
                config.apiBaseUrl(), cleanSlug, filename);

        LOG.infof("Downloading: %s from Kaggle model %s", filename, cleanSlug);

        HttpResponse<InputStream> response = httpClient.send(
                buildGetRequest(url),
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() == 401 || response.statusCode() == 403) {
            throw new IOException(String.format(
                    "Kaggle authentication failed downloading %s (HTTP %d). Check Kaggle credentials.",
                    filename, response.statusCode()));
        }

        if (response.statusCode() != 200) {
            throw new IOException(String.format(
                    "Failed to download file '%s' from Kaggle: HTTP %d", filename, response.statusCode()));
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

        resolveCredentials().ifPresent(creds -> {
            if (!creds.username().isBlank()) {
                String auth = creds.username() + ":" + creds.key();
                String encoded = java.util.Base64.getEncoder().encodeToString(
                        auth.getBytes(java.nio.charset.StandardCharsets.UTF_8));
                builder.header("Authorization", "Basic " + encoded);
            } else {
                builder.header("Authorization", "Bearer " + creds.key());
            }
        });
        return builder.build();
    }

    public record KaggleCredentials(String username, String key) {}

    public Optional<KaggleCredentials> resolveCredentials() {
        if (config != null && config.username().isPresent() && config.token().isPresent()) {
            String u = config.username().get().trim();
            String k = config.token().get().trim();
            if (!u.isBlank() && !k.isBlank() && !u.contains("${") && !k.contains("${")) {
                return Optional.of(new KaggleCredentials(u, k));
            }
        }

        String envUser = System.getenv("KAGGLE_USERNAME");
        String envKey = System.getenv("KAGGLE_KEY");
        if (envUser != null && !envUser.isBlank() && envKey != null && !envKey.isBlank()) {
            return Optional.of(new KaggleCredentials(envUser.trim(), envKey.trim()));
        }

        String envToken = System.getenv("KAGGLE_API_TOKEN");
        if (envToken == null || envToken.isBlank()) envToken = System.getenv("KAGGLE_TOKEN");
        if (envToken != null && !envToken.isBlank()) {
            return Optional.of(new KaggleCredentials("", envToken.trim()));
        }

        try {
            Path kaggleJson = Path.of(System.getProperty("user.home"), ".kaggle", "kaggle.json");
            if (Files.exists(kaggleJson)) {
                String content = Files.readString(kaggleJson);
                var node = objectMapper.readTree(content);
                if (node.has("username") && node.has("key")) {
                    return Optional.of(new KaggleCredentials(node.get("username").asText(), node.get("key").asText()));
                }
            }
        } catch (Exception ignored) {}

        return Optional.empty();
    }

    private String cleanSlug(String raw) {
        if (raw == null) return "";
        String s = raw.trim();
        if (s.startsWith("kg:")) s = s.substring(3).trim();
        if (s.startsWith("kaggle:")) s = s.substring(7).trim();
        if (s.startsWith("kaggle://")) s = s.substring(9).trim();
        return s;
    }

    // ── Inner records for JSON parsing ──────────────────────────────────

    public record KaggleFileList(List<KaggleFileEntry> files) {}
    public record KaggleFileEntry(String path, long size) {}
}
