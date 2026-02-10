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

package tech.kayys.golek.model.download;

/**
 * Progress listener for downloads
 */
public interface DownloadProgressListener {

    /**
     * Called when progress is made
     * 
     * @param downloadedBytes bytes downloaded so far
     * @param totalBytes      total bytes to download (-1 if unknown)
     * @param progress        progress as decimal (0.0 to 1.0)
     */
    void onProgress(long downloadedBytes, long totalBytes, double progress);

    /**
     * Called when download starts
     */
    default void onStart(long totalBytes) {
    }

    /**
     * Called when download completes
     */
    default void onComplete(long totalBytes) {
    }

    /**
     * Called when download fails
     */
    default void onError(Throwable error) {
    }
}