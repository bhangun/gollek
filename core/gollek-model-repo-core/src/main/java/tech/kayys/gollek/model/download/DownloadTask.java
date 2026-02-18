/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */

package tech.kayys.gollek.model.download;

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
