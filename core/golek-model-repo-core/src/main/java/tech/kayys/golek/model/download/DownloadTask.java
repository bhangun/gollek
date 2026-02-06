package tech.kayys.golek.model.download;

import java.nio.file.Path;
import java.util.UUID;

/**
 * Represents a download task and its state
 */
public class DownloadTask {
    private final String id;
    private final String uri;
    private final Path targetPath;
    private long totalBytes;
    private long downloadedBytes;
    private DownloadStatus status;
    private String checksum;

    public DownloadTask(String uri, Path targetPath) {
        this.id = UUID.randomUUID().toString();
        this.uri = uri;
        this.targetPath = targetPath;
        this.status = DownloadStatus.PENDING;
    }

    // Getters and Setters

    public String getId() {
        return id;
    }

    public String getUri() {
        return uri;
    }

    public Path getTargetPath() {
        return targetPath;
    }

    public long getTotalBytes() {
        return totalBytes;
    }

    public void setTotalBytes(long totalBytes) {
        this.totalBytes = totalBytes;
    }

    public long getDownloadedBytes() {
        return downloadedBytes;
    }

    public void setDownloadedBytes(long downloadedBytes) {
        this.downloadedBytes = downloadedBytes;
    }

    public DownloadStatus getStatus() {
        return status;
    }

    public void setStatus(DownloadStatus status) {
        this.status = status;
    }

    public String getChecksum() {
        return checksum;
    }

    public void setChecksum(String checksum) {
        this.checksum = checksum;
    }

    public double getProgress() {
        if (totalBytes <= 0)
            return 0;
        return (double) downloadedBytes / totalBytes;
    }

    public enum DownloadStatus {
        PENDING, DOWNLOADING, PAUSED, COMPLETED, FAILED
    }
}
