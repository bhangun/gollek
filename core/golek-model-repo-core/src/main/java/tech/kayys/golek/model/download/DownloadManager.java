package tech.kayys.golek.model.download;

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
            DownloadProgressListener listener) {
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
            DownloadProgressListener listener) throws IOException {

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