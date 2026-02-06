package tech.kayys.golek.model.download;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;

/**
 * Advanced DownloadManager with parallel chunk support and resumability
 */
@ApplicationScoped
public class DownloadManager {

    private static final Logger LOG = Logger.getLogger(DownloadManager.class);
    private static final int DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024; // 5MB chunks
    private static final int MAX_PARALLEL_CHUNKS = 4;

    private final ExecutorService executor = Executors.newFixedThreadPool(MAX_PARALLEL_CHUNKS * 2);

    /**
     * Download a file in parallel using a range-providing function
     */
    public CompletionStage<Path> downloadParallel(
            String uri,
            Path targetPath,
            long totalBytes,
            BiFunction<Long, Long, InputStream> rangeProvider,
            DownloadProgressListener listener) {

        return CompletableFuture.supplyAsync(() -> {
            try {
                return doDownloadParallel(uri, targetPath, totalBytes, rangeProvider, listener);
            } catch (Exception e) {
                if (listener != null)
                    listener.onError(e);
                throw new RuntimeException("Parallel download failed", e);
            }
        }, executor);
    }

    private Path doDownloadParallel(
            String uri,
            Path targetPath,
            long totalBytes,
            BiFunction<Long, Long, InputStream> rangeProvider,
            DownloadProgressListener listener) throws IOException, InterruptedException, ExecutionException {

        if (listener != null)
            listener.onStart(totalBytes);

        Files.createDirectories(targetPath.getParent());
        Path tempPath = targetPath.resolveSibling(targetPath.getFileName() + ".downloading");

        // Prepare file size
        try (FileChannel fc = FileChannel.open(tempPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            if (totalBytes > 0) {
                fc.position(totalBytes - 1);
                fc.write(ByteBuffer.wrap(new byte[] { 0 }));
            }
        }

        int chunkCount = totalBytes > 0 ? (int) Math.ceil((double) totalBytes / DEFAULT_CHUNK_SIZE) : 1;
        AtomicLong totalDownloaded = new AtomicLong(0);
        List<CompletableFuture<Void>> futures = new ArrayList<>();

        LOG.infof("Starting parallel download for %s (%d chunks)", uri, chunkCount);

        for (int i = 0; i < chunkCount; i++) {
            long start = (long) i * DEFAULT_CHUNK_SIZE;
            long end = totalBytes > 0 ? Math.min(start + DEFAULT_CHUNK_SIZE - 1, totalBytes - 1) : -1;

            futures.add(CompletableFuture.runAsync(() -> {
                try (InputStream is = rangeProvider.apply(start, end);
                        FileChannel fc = FileChannel.open(tempPath, StandardOpenOption.WRITE)) {

                    long position = start;
                    int bytesRead;
                    byte[] bytes = new byte[8192];

                    while ((bytesRead = is.read(bytes)) != -1) {
                        fc.write(ByteBuffer.wrap(bytes, 0, bytesRead), position);
                        position += bytesRead;
                        long currentTotal = totalDownloaded.addAndGet(bytesRead);

                        if (listener != null && totalBytes > 0) {
                            listener.onProgress(currentTotal, totalBytes, (double) currentTotal / totalBytes);
                        }
                    }
                } catch (IOException e) {
                    throw new RuntimeException("Chunk download failed", e);
                }
            }, executor));

            if (futures.size() >= MAX_PARALLEL_CHUNKS) {
                CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();
                futures.clear();
            }
        }

        if (!futures.isEmpty()) {
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();
        }

        Files.move(tempPath, targetPath, StandardCopyOption.REPLACE_EXISTING);

        if (listener != null)
            listener.onComplete(totalDownloaded.get());
        LOG.infof("Parallel download complete: %s", targetPath);
        return targetPath;
    }

    public void shutdown() {
        executor.shutdown();
    }
}