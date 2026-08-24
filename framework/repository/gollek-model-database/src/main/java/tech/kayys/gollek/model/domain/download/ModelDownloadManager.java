package tech.kayys.gollek.model.domain.download;

import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tech.kayys.alkhawarizm.spi.download.DownloadProgressListener;
import tech.kayys.gollek.model.download.DownloadManager; // The core one

@ApplicationScoped
public class ModelDownloadManager {

    private static final Logger LOG = LoggerFactory.getLogger(ModelDownloadManager.class);

    @Inject
    DownloadManager coreDownloadManager;

    private final Map<String, ModelDownloadState> activeDownloads = new ConcurrentHashMap<>();
    
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .followRedirects(HttpClient.Redirect.NORMAL)
            .build();

    public ModelDownloadState getDownloadState(String modelId) {
        return activeDownloads.get(modelId);
    }

    public Map<String, ModelDownloadState> getAllActiveDownloads() {
        return activeDownloads;
    }

    public CompletableFuture<Path> startDownload(String modelId, String uriString) {
        LOG.info("Starting download for model: {} from URI: {}", modelId, uriString);
        
        DownloadProgressListener listener = createListener(modelId);
        
        return CompletableFuture.supplyAsync(() -> {
            try {
                URI uri = URI.create(uriString);
                
                // 1. Get file size via HEAD
                HttpRequest headRequest = HttpRequest.newBuilder(uri)
                        .method("HEAD", HttpRequest.BodyPublishers.noBody())
                        .build();
                        
                HttpResponse<Void> headResponse = httpClient.send(headRequest, HttpResponse.BodyHandlers.discarding());
                long totalBytes = -1;
                var contentLengthHeader = headResponse.headers().firstValue("Content-Length");
                if (contentLengthHeader.isPresent()) {
                    totalBytes = Long.parseLong(contentLengthHeader.get());
                }
                
                LOG.info("Model {} size: {} bytes", modelId, totalBytes);
                
                // 2. Setup path
                String userHome = System.getProperty("user.home");
                Path targetPath = Paths.get(userHome, ".gollek", "models", modelId);
                
                // 3. Define Range Provider
                BiFunction<Long, Long, InputStream> rangeProvider = (start, end) -> {
                    try {
                        HttpRequest.Builder reqBuilder = HttpRequest.newBuilder(uri).GET();
                        if (start >= 0 && end > 0) {
                            reqBuilder.header("Range", "bytes=" + start + "-" + end);
                        }
                        
                        HttpResponse<InputStream> response = httpClient.send(reqBuilder.build(), HttpResponse.BodyHandlers.ofInputStream());
                        if (response.statusCode() >= 400) {
                            throw new RuntimeException("HTTP error " + response.statusCode());
                        }
                        return response.body();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                };
                
                // 4. Start Download
                return coreDownloadManager.downloadParallel(uriString, targetPath, totalBytes, rangeProvider, listener)
                        .toCompletableFuture()
                        .get();
                        
            } catch (Exception e) {
                LOG.error("Failed to download model: {}", modelId, e);
                listener.onError(e);
                throw new RuntimeException("Download failed", e);
            }
        });
    }

    public DownloadProgressListener createListener(String modelId) {
        ModelDownloadState state = new ModelDownloadState(modelId);
        activeDownloads.put(modelId, state);

        return new DownloadProgressListener() {
            @Override
            public void onStart(long totalBytes) {
                state.setTotalBytes(totalBytes);
                state.setStatus("DOWNLOADING");
            }

            @Override
            public void onProgress(long bytesDownloaded, long totalBytes, double percentage) {
                state.setBytesDownloaded(bytesDownloaded);
                state.setPercentage(percentage);
            }

            @Override
            public void onComplete(long totalBytes) {
                state.setBytesDownloaded(totalBytes);
                state.setPercentage(1.0);
                state.setStatus("COMPLETED");
                LOG.info("Download complete for model: {}", modelId);
            }

            @Override
            public void onError(Throwable e) {
                state.setStatus("FAILED");
                state.setErrorMessage(e.getMessage());
            }
        };
    }
}
