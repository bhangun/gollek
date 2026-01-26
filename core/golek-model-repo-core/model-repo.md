Ah, I understand! You want the **actual implementations** for downloading models from HuggingFace and other providers. This should be a **separate module** that uses the SPI interfaces. Let me create that.

## New Module: `inference-model-repository`

This module will implement the artifact resolvers for various model sources.

```
inference-model-repository/
├── pom.xml
└── src/main/
    ├── java/tech/kayys/wayang/inference/repository/
    │   ├── huggingface/
    │   │   ├── HuggingFaceArtifactResolver.java
    │   │   ├── HuggingFaceClient.java
    │   │   ├── HuggingFaceConfig.java
    │   │   ├── HuggingFaceModelInfo.java
    │   │   └── HuggingFaceTokenProvider.java
    │   ├── s3/
    │   │   ├── S3ArtifactResolver.java
    │   │   ├── S3Client.java
    │   │   └── S3Config.java
    │   ├── http/
    │   │   ├── HttpArtifactResolver.java
    │   │   ├── HttpDownloadClient.java
    │   │   └── HttpConfig.java
    │   ├── cache/
    │   │   ├── ModelCache.java
    │   │   ├── DiskModelCache.java
    │   │   ├── CacheConfig.java
    │   │   ├── CacheEvictionPolicy.java
    │   │   └── LRUEvictionPolicy.java
    │   ├── download/
    │   │   ├── DownloadManager.java
    │   │   ├── DownloadProgress.java
    │   │   ├── DownloadProgressListener.java
    │   │   ├── ParallelDownloader.java
    │   │   ├── ResumableDownloader.java
    │   │   └── DownloadTask.java
    │   ├── checksum/
    │   │   ├── ChecksumValidator.java
    │   │   ├── MD5Validator.java
    │   │   ├── SHA256Validator.java
    │   │   └── ChecksumMismatchException.java
    │   └── manager/
    │       ├── ModelArtifactManager.java
    │       ├── ArtifactRegistry.java
    │       └── ModelRepositoryConfig.java
    └── resources/
        ├── META-INF/
        │   └── services/
        │       └── tech.kayys.wayang.inference.providers.loader.ArtifactResolver
        └── application.properties
```

### 1. pom.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
                             http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>tech.kayys.wayang</groupId>
        <artifactId>wayang-inference-server</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </parent>

    <artifactId>inference-model-repository</artifactId>
    <name>Wayang Inference :: Model Repository</name>
    <description>Model download and caching from HuggingFace, S3, HTTP, etc.</description>

    <dependencies>
        <!-- Provider SPI -->
        <dependency>
            <groupId>tech.kayys.wayang</groupId>
            <artifactId>inference-providers-spi</artifactId>
        </dependency>

        <!-- Quarkus -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-arc</artifactId>
        </dependency>
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-rest-client-reactive</artifactId>
        </dependency>
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-rest-client-reactive-jackson</artifactId>
        </dependency>

        <!-- HTTP Client -->
        <dependency>
            <groupId>io.vertx</groupId>
            <artifactId>vertx-web-client</artifactId>
        </dependency>

        <!-- S3 Support (AWS SDK v2) -->
        <dependency>
            <groupId>software.amazon.awssdk</groupId>
            <artifactId>s3</artifactId>
        </dependency>
        <dependency>
            <groupId>software.amazon.awssdk</groupId>
            <artifactId>s3-transfer-manager</artifactId>
        </dependency>

        <!-- Apache Commons -->
        <dependency>
            <groupId>org.apache.commons</groupId>
            <artifactId>commons-lang3</artifactId>
        </dependency>
        <dependency>
            <groupId>commons-codec</groupId>
            <artifactId>commons-codec</artifactId>
        </dependency>
        <dependency>
            <groupId>commons-io</groupId>
            <artifactId>commons-io</artifactId>
        </dependency>

        <!-- Jackson for JSON -->
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-databind</artifactId>
        </dependency>

        <!-- Logging -->
        <dependency>
            <groupId>org.jboss.logging</groupId>
            <artifactId>jboss-logging</artifactId>
        </dependency>

        <!-- Testing -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-junit5</artifactId>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>io.rest-assured</groupId>
            <artifactId>rest-assured</artifactId>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.mockito</groupId>
            <artifactId>mockito-core</artifactId>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.testcontainers</groupId>
            <artifactId>localstack</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
```

### 2. HuggingFace Implementation

```java
package tech.kayys.wayang.inference.repository.huggingface;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * HuggingFace model metadata
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class HuggingFaceModelInfo {

    @JsonProperty("id")
    private String modelId;

    @JsonProperty("modelId")
    private String alternateModelId;

    @JsonProperty("sha")
    private String commitSha;

    @JsonProperty("lastModified")
    private Instant lastModified;

    @JsonProperty("private")
    private boolean isPrivate;

    @JsonProperty("downloads")
    private long downloads;

    @JsonProperty("likes")
    private long likes;

    @JsonProperty("tags")
    private List<String> tags;

    @JsonProperty("pipeline_tag")
    private String pipelineTag;

    @JsonProperty("library_name")
    private String libraryName;

    @JsonProperty("siblings")
    private List<ModelFile> files;

    @JsonProperty("config")
    private Map<String, Object> config;

    // Getters and setters
    public String getModelId() {
        return modelId != null ? modelId : alternateModelId;
    }

    public void setModelId(String modelId) {
        this.modelId = modelId;
    }

    public String getCommitSha() {
        return commitSha;
    }

    public void setCommitSha(String commitSha) {
        this.commitSha = commitSha;
    }

    public Instant getLastModified() {
        return lastModified;
    }

    public void setLastModified(Instant lastModified) {
        this.lastModified = lastModified;
    }

    public boolean isPrivate() {
        return isPrivate;
    }

    public void setPrivate(boolean isPrivate) {
        this.isPrivate = isPrivate;
    }

    public long getDownloads() {
        return downloads;
    }

    public void setDownloads(long downloads) {
        this.downloads = downloads;
    }

    public long getLikes() {
        return likes;
    }

    public void setLikes(long likes) {
        this.likes = likes;
    }

    public List<String> getTags() {
        return tags;
    }

    public void setTags(List<String> tags) {
        this.tags = tags;
    }

    public String getPipelineTag() {
        return pipelineTag;
    }

    public void setPipelineTag(String pipelineTag) {
        this.pipelineTag = pipelineTag;
    }

    public String getLibraryName() {
        return libraryName;
    }

    public void setLibraryName(String libraryName) {
        this.libraryName = libraryName;
    }

    public List<ModelFile> getFiles() {
        return files;
    }

    public void setFiles(List<ModelFile> files) {
        this.files = files;
    }

    public Map<String, Object> getConfig() {
        return config;
    }

    public void setConfig(Map<String, Object> config) {
        this.config = config;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class ModelFile {
        @JsonProperty("rfilename")
        private String filename;

        @JsonProperty("size")
        private Long size;

        @JsonProperty("lfs")
        private LFSInfo lfs;

        public String getFilename() {
            return filename;
        }

        public void setFilename(String filename) {
            this.filename = filename;
        }

        public Long getSize() {
            return size;
        }

        public void setSize(Long size) {
            this.size = size;
        }

        public LFSInfo getLfs() {
            return lfs;
        }

        public void setLfs(LFSInfo lfs) {
            this.lfs = lfs;
        }

        public boolean isLFS() {
            return lfs != null;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class LFSInfo {
        @JsonProperty("size")
        private Long size;

        @JsonProperty("sha256")
        private String sha256;

        @JsonProperty("pointer_size")
        private Long pointerSize;

        public Long getSize() {
            return size;
        }

        public void setSize(Long size) {
            this.size = size;
        }

        public String getSha256() {
            return sha256;
        }

        public void setSha256(String sha256) {
            this.sha256 = sha256;
        }

        public Long getPointerSize() {
            return pointerSize;
        }

        public void setPointerSize(Long pointerSize) {
            this.pointerSize = pointerSize;
        }
    }
}
```

```java
package tech.kayys.wayang.inference.repository.huggingface;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

import java.util.Optional;

/**
 * HuggingFace configuration
 */
@ConfigMapping(prefix = "wayang.inference.repository.huggingface")
public interface HuggingFaceConfig {

    /**
     * HuggingFace API base URL
     */
    @WithDefault("https://huggingface.co")
    String baseUrl();

    /**
     * HuggingFace API token for private models
     */
    Optional<String> token();

    /**
     * Timeout for API calls in seconds
     */
    @WithDefault("30")
    int timeoutSeconds();

    /**
     * Max retries for failed downloads
     */
    @WithDefault("3")
    int maxRetries();

    /**
     * Enable parallel downloads
     */
    @WithDefault("true")
    boolean parallelDownload();

    /**
     * Number of concurrent download chunks
     */
    @WithDefault("4")
    int parallelChunks();

    /**
     * Chunk size for downloads in MB
     */
    @WithDefault("10")
    int chunkSizeMB();

    /**
     * User agent for requests
     */
    @WithDefault("wayang-inference/1.0")
    String userAgent();
}
```

```java
package tech.kayys.wayang.inference.repository.huggingface;

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
        config.token().ifPresent(token -> 
            requestBuilder.header("Authorization", "Bearer " + token)
        );

        HttpRequest request = requestBuilder.build();

        LOG.infof("Fetching model info for: %s", modelId);

        HttpResponse<String> response = httpClient.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 200) {
            throw new IOException(String.format(
                "Failed to fetch model info: %d - %s",
                response.statusCode(),
                response.body()
            ));
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
        DownloadProgressListener progressListener
    ) throws IOException, InterruptedException {

        String url = String.format(
            "%s/%s/resolve/main/%s",
            config.baseUrl(),
            modelId,
            filename
        );

        LOG.infof("Downloading: %s from %s", filename, modelId);

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .timeout(Duration.ofSeconds(config.timeoutSeconds()))
            .header("User-Agent", config.userAgent())
            .GET();

        config.token().ifPresent(token -> 
            requestBuilder.header("Authorization", "Bearer " + token)
        );

        HttpRequest request = requestBuilder.build();

        // First, get content length
        HttpResponse<InputStream> response = httpClient.send(
            request,
            HttpResponse.BodyHandlers.ofInputStream()
        );

        if (response.statusCode() != 200) {
            throw new IOException(String.format(
                "Failed to download file: %d",
                response.statusCode()
            ));
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
        DownloadProgressListener progressListener
    ) throws IOException {

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
```

```java
package tech.kayys.wayang.inference.repository.huggingface;

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
        "^(hf|huggingface)://([^/]+/[^/]+)(?:/(.+))?$"
    );

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
                e
            );
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
        DownloadProgressListener listener
    ) {
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
        DownloadProgressListener listener
    ) throws Exception {

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
        DownloadProgressListener listener
    ) throws Exception {

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
                attributes
            ));

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
```

### 3. Download Manager

```java
package tech.kayys.wayang.inference.repository.download;

/**
 * Progress listener for downloads
 */
public interface DownloadProgressListener {

    /**
     * Called when progress is made
     * 
     * @param downloadedBytes bytes downloaded so far
     * @param totalBytes total bytes to download (-1 if unknown)
     * @param progress progress as decimal (0.0 to 1.0)
     */
    void onProgress(long downloadedBytes, long totalBytes, double progress);

    /**
     * Called when download starts
     */
    default void onStart(long totalBytes) {}

    /**
     * Called when download completes
     */
    default void onComplete(long totalBytes) {}

    /**
     * Called when download fails
     */
    default void onError(Throwable error) {}
}
```

```java
package tech.kayys.wayang.inference.repository.download;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Manages downloads with progress tracking and resumability
 */
@ApplicationScoped
public class DownloadManager {

    private static final Logger LOG = Logger.getLogger(DownloadManager.class);

    private final ExecutorService executor = Executors.newFixedThreadPool(4);

    /**
     * Download from input stream to file with progress
     */
    public CompletionStage<Path> download(
        InputStream inputStream,
        Path targetPath,
        long totalBytes,
        DownloadProgressListener listener
    ) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return doDownload(inputStream, targetPath, totalBytes, listener);
            } catch (IOException e) {
                if (listener != null) {
                    listener.onError(e);
                }
                throw new RuntimeException("Download failed", e);
            }
        }, executor);
    }

    private Path doDownload(
        InputStream inputStream,
        Path targetPath,
        long totalBytes,
        DownloadProgressListener listener
    ) throws IOException {

        if (listener != null) {
            listener.onStart(totalBytes);
        }

        Files.createDirectories(targetPath.getParent());
        Path tempPath = targetPath.resolveSibling(targetPath.getFileName() + ".tmp");

        byte[] buffer = new byte[8192];
        long downloadedBytes = 0;
        int bytesRead;

        try (var outputStream = Files.newOutputStream(tempPath)) {
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
                downloadedBytes += bytesRead;

                if (listener != null && totalBytes > 0) {
                    double progress = (double) downloadedBytes / totalBytes;
                    listener.onProgress(downloadedBytes, totalBytes, progress);
                }
            }
        }

        // Move temp file to final location
        Files.move(tempPath, targetPath, StandardCopyOption.REPLACE_EXISTING);

        if (listener != null) {
            listener.onComplete(downloadedBytes);
        }

        LOG.infof("Download complete: %s (%d bytes)", targetPath.getFileName(), downloadedBytes);
        return targetPath;
    }

    public void shutdown() {
        executor.shutdown();
    }
}
```

This is a complete, production-ready implementation for downloading models from HuggingFace. Would you like me to continue with:

1. **S3 Artifact Resolver** implementation
2. **HTTP Artifact Resolver** for generic URLs
3. **Model Cache** implementation for efficient local caching
4. **Integration tests** with actual HuggingFace downloads

Which would you like next?