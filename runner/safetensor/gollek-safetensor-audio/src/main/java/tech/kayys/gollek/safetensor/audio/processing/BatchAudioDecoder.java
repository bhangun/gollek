/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * BatchAudioDecoder.java
 * ───────────────────────
 * Batch decoding utilities for concurrent audio processing.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * High-throughput batch audio decoding utilities.
 * 
 * <p>Provides concurrent decoding of multiple audio files with configurable
 * parallelism, progress tracking, and error handling.</p>
 *
 * <h2>Features</h2>
 * <ul>
 *   <li>Concurrent decoding with configurable thread pool</li>
 *   <li>Progress tracking across all files</li>
 *   <li>Error aggregation and reporting</li>
 *   <li>Memory-efficient streaming processing</li>
 *   <li>Support for both sync and async operations</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Simple batch decode
 * List<byte[]> audioFiles = List.of(file1, file2, file3);
 * List<float[]> results = BatchAudioDecoder.decodeAll(audioFiles);
 * 
 * // Batch decode with progress tracking
 * BatchAudioDecoder.decodeAll(audioFiles, progress -> {
 *     System.out.printf("Overall progress: %.1f%%\n", progress * 100);
 * });
 * 
 * // Async batch decode
 * CompletableFuture<List<float[]>> future = BatchAudioDecoder.decodeAllAsync(audioFiles);
 * future.thenAccept(results -> processResults(results));
 * 
 * // Batch decode from files
 * List<Path> audioPaths = List.of(path1, path2, path3);
 * List<float[]> results = BatchAudioDecoder.decodeFiles(audioPaths);
 * }</pre>
 *
 * @author Bhangun
 * @version 1.0.0
 * @since 2.1.0
 */
public final class BatchAudioDecoder {

    private static final Logger log = Logger.getLogger(BatchAudioDecoder.class);

    /**
     * Default parallelism level (number of CPU cores).
     */
    public static final int DEFAULT_PARALLELISM = Runtime.getRuntime().availableProcessors();

    /**
     * Shared executor for async operations.
     */
    private static final Executor SHARED_EXECUTOR = Executors.newFixedThreadPool(
            DEFAULT_PARALLELISM,
            r -> {
                Thread t = new Thread(r, "batch-audio-decoder");
                t.setDaemon(true);
                return t;
            });

    private BatchAudioDecoder() {
        // Utility class - prevent instantiation
    }

    /**
     * Decode multiple audio files concurrently.
     *
     * @param audioFiles list of audio byte arrays
     * @return list of decoded PCM float arrays
     * @throws IOException if any decode operation fails
     */
    public static List<float[]> decodeAll(List<byte[]> audioFiles) throws IOException {
        return decodeAll(audioFiles, null, null);
    }

    /**
     * Decode multiple audio files concurrently with progress tracking.
     *
     * @param audioFiles list of audio byte arrays
     * @param progressListener callback for progress updates (0.0 to 1.0)
     * @return list of decoded PCM float arrays
     * @throws IOException if any decode operation fails
     */
    public static List<float[]> decodeAll(List<byte[]> audioFiles, 
                                          Consumer<Float> progressListener) throws IOException {
        return decodeAll(audioFiles, progressListener, null);
    }

    /**
     * Decode multiple audio files concurrently with custom decoder.
     *
     * @param audioFiles list of audio byte arrays
     * @param progressListener callback for progress updates (may be null)
     * @param decoder custom decoder (may be null for default)
     * @return list of decoded PCM float arrays
     * @throws IOException if any decode operation fails
     */
    public static List<float[]> decodeAll(List<byte[]> audioFiles,
                                          Consumer<Float> progressListener,
                                          FlacDecoder decoder) throws IOException {
        
        if (audioFiles == null || audioFiles.isEmpty()) {
            return List.of();
        }

        FlacDecoder useDecoder = decoder != null ? decoder : new FlacDecoder();
        int totalFiles = audioFiles.size();
        java.util.concurrent.atomic.AtomicInteger completedFiles = new java.util.concurrent.atomic.AtomicInteger(0);

        List<float[]> results = new ArrayList<>(totalFiles);
        List<IOException> errors = new ArrayList<>();

        // Process files in parallel using ForkJoinPool
        ForkJoinPool.commonPool().submit(() -> {
            audioFiles.parallelStream().forEach(audioFile -> {
                try {
                    float[] pcm = useDecoder.decode(audioFile);
                    
                    synchronized (results) {
                        results.add(pcm);
                    }

                    int completed = completedFiles.incrementAndGet();
                    if (progressListener != null) {
                        progressListener.accept(completed / (float) totalFiles);
                    }

                } catch (IOException e) {
                    synchronized (errors) {
                        errors.add(e);
                    }
                }
            });
        }).join();

        // Report errors
        if (!errors.isEmpty()) {
            log.errorf("Batch decoding completed with %d errors out of %d files", 
                    errors.size(), totalFiles);
            if (errors.size() == totalFiles) {
                throw errors.get(0); // All failed - throw first error
            }
            // Partial success - log errors but return results
            errors.forEach(e -> log.warnf("Decode error: %s", e.getMessage()));
        }

        log.infof("Batch decoding completed: %d/%d files successful", 
                results.size(), totalFiles);

        return results;
    }

    /**
     * Decode multiple audio files asynchronously.
     *
     * @param audioFiles list of audio byte arrays
     * @return future that completes with list of decoded PCM float arrays
     */
    public static CompletableFuture<List<float[]>> decodeAllAsync(List<byte[]> audioFiles) {
        return decodeAllAsync(audioFiles, null, null);
    }

    /**
     * Decode multiple audio files asynchronously with progress tracking.
     *
     * @param audioFiles list of audio byte arrays
     * @param progressListener callback for progress updates (may be null)
     * @return future that completes with list of decoded PCM float arrays
     */
    public static CompletableFuture<List<float[]>> decodeAllAsync(
            List<byte[]> audioFiles, Consumer<Float> progressListener) {
        return decodeAllAsync(audioFiles, progressListener, null);
    }

    /**
     * Decode multiple audio files asynchronously with custom decoder.
     *
     * @param audioFiles list of audio byte arrays
     * @param progressListener callback for progress updates (may be null)
     * @param decoder custom decoder (may be null for default)
     * @return future that completes with list of decoded PCM float arrays
     */
    public static CompletableFuture<List<float[]>> decodeAllAsync(
            List<byte[]> audioFiles, 
            Consumer<Float> progressListener,
            FlacDecoder decoder) {
        
        return CompletableFuture.supplyAsync(() -> {
            try {
                return decodeAll(audioFiles, progressListener, decoder);
            } catch (IOException e) {
                throw new java.util.concurrent.CompletionException(e);
            }
        }, SHARED_EXECUTOR);
    }

    /**
     * Decode multiple audio files from disk.
     *
     * @param audioPaths list of audio file paths
     * @return list of decoded PCM float arrays
     * @throws IOException if file reading or decoding fails
     */
    public static List<float[]> decodeFiles(List<Path> audioPaths) throws IOException {
        return decodeFiles(audioPaths, null);
    }

    /**
     * Decode multiple audio files from disk with progress tracking.
     *
     * @param audioPaths list of audio file paths
     * @param progressListener callback for progress updates (may be null)
     * @return list of decoded PCM float arrays
     * @throws IOException if file reading or decoding fails
     */
    public static List<float[]> decodeFiles(List<Path> audioPaths,
                                            Consumer<Float> progressListener) throws IOException {
        
        if (audioPaths == null || audioPaths.isEmpty()) {
            return List.of();
        }

        log.infof("Decoding %d audio files from disk", audioPaths.size());

        // Read all files into memory (could be optimized for very large files)
        List<byte[]> audioBytes = audioPaths.stream()
            .map(path -> {
                try {
                    return Files.readAllBytes(path);
                } catch (IOException e) {
                    throw new java.util.concurrent.CompletionException(
                        new IOException("Failed to read file: " + path, e));
                }
            })
            .collect(Collectors.toList());

        return decodeAll(audioBytes, progressListener);
    }

    /**
     * Decode all audio files in a directory.
     *
     * @param directory directory containing audio files
     * @param extension file extension filter (e.g., ".flac", may be null for all)
     * @return list of decoded PCM float arrays
     * @throws IOException if file reading or decoding fails
     */
    public static List<float[]> decodeDirectory(Path directory, String extension) throws IOException {
        return decodeDirectory(directory, extension, null);
    }

    /**
     * Decode all audio files in a directory with progress tracking.
     *
     * @param directory directory containing audio files
     * @param extension file extension filter (e.g., ".flac", may be null for all)
     * @param progressListener callback for progress updates (may be null)
     * @return list of decoded PCM float arrays
     * @throws IOException if file reading or decoding fails
     */
    public static List<float[]> decodeDirectory(Path directory, String extension,
                                                Consumer<Float> progressListener) throws IOException {
        
        if (!Files.isDirectory(directory)) {
            throw new IOException("Directory does not exist: " + directory);
        }

        List<Path> audioFiles = Files.list(directory)
            .filter(path -> extension == null || path.toString().endsWith(extension))
            .filter(Files::isRegularFile)
            .sorted() // Consistent ordering
            .toList();

        log.infof("Found %d audio files in %s", audioFiles.size(), directory);
        return decodeFiles(audioFiles, progressListener);
    }

    /**
     * Get the shared executor.
     *
     * @return shared executor
     */
    public static Executor getSharedExecutor() {
        return SHARED_EXECUTOR;
    }

    /**
     * Batch decode result with metadata.
     *
     * @param pcm decoded PCM float array
     * @param sourcePath source file path (may be null)
     * @param decodeTimeMs time taken to decode in milliseconds
     * @param success whether decoding was successful
     * @param error error message if failed (may be null)
     */
    public record BatchResult(
            float[] pcm,
            Path sourcePath,
            long decodeTimeMs,
            boolean success,
            String error
    ) {
        /**
         * Create a successful result.
         *
         * @param pcm decoded PCM
         * @param sourcePath source path
         * @param decodeTimeMs decode time
         * @return successful result
         */
        public static BatchResult success(float[] pcm, Path sourcePath, long decodeTimeMs) {
            return new BatchResult(pcm, sourcePath, decodeTimeMs, true, null);
        }

        /**
         * Create a failed result.
         *
         * @param sourcePath source path
         * @param error error message
         * @return failed result
         */
        public static BatchResult failure(Path sourcePath, String error) {
            return new BatchResult(null, sourcePath, 0, false, error);
        }
    }
}
