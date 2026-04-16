/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorLoaderFacade.java
 * ───────────────────────────
 * High-level facade that integrates the FFM loader, shard loader, and LRU
 * cache behind a single CDI bean.
 *
 * This is the recommended injection point for all consumers outside the
 * safetensor module (e.g. SafetensorProvider, test harnesses, REST endpoints).
 *
 * It provides:
 *  • Transparent single-file / sharded model routing
 *  • LRU caching of open load results
 *  • Structured error mapping to ProviderException for Gollek SPI consumers
 *  • Reactive (Mutiny) wrappers for integration with async request pipelines
 *  • Header-only inspection for lightweight model catalogue queries
 */
package tech.kayys.gollek.safetensor.loader;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.safetensor.exception.SafetensorException;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Primary facade for SafeTensors loading — the recommended injection point.
 *
 * <p>
 * Hides the complexity of single-file vs sharded models, mmap vs copy
 * mode, LRU caching, and error translation.
 *
 * <pre>{@code
 * @Inject
 * SafetensorLoaderFacade safetensorLoader;
 *
 * // Simple synchronous access (blocking)
 * try (SafetensorShardSession s = safetensorLoader.open(Path.of("/models/llama3"))) {
 *     float[] weights = s.tensor("lm_head.weight").toFloatArray();
 * }
 *
 * // Reactive access
 * Uni<float[]> weights = safetensorLoader.loadTensorAsync(
 *         Path.of("/models/llama3"), "lm_head.weight",
 *         tensor -> tensor.toFloatArray());
 * }</pre>
 */
@ApplicationScoped
public class SafetensorLoaderFacade {

    private static final Logger log = Logger.getLogger(SafetensorLoaderFacade.class);

    private static final String PROVIDER_ID = "safetensor";

    @Inject
    SafetensorFFMLoader ffmLoader;

    @Inject
    SafetensorShardLoader shardLoader;

    @Inject
    SafetensorLoadCache loadCache;

    // ─────────────────────────────────────────────────────────────────────────
    // Session API (recommended for multi-tensor access)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Open a SafeTensors model at the given path.
     *
     * <p>
     * Handles both single-file and multi-shard layouts transparently.
     * The caller is responsible for closing the returned session.
     *
     * @param modelPath path to a .safetensors file or model directory
     * @return open session (must be closed by the caller)
     * @throws ProviderException on any loading failure
     */
    public SafetensorShardSession open(Path modelPath) {
        log.info("Opening SafeTensors model at " + modelPath);
        try {
            return shardLoader.open(modelPath);
        } catch (SafetensorException e) {
            throw wrapProviderException("Failed to open SafeTensors model at " + modelPath, e, true);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Direct (cached) single-file API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Load (or retrieve from cache) a single-file SafeTensors result.
     *
     * <p>
     * The result is managed by the LRU cache — do NOT call {@code close()} on
     * it; the cache handles eviction. If you need explicit lifetime control,
     * use {@link #open(Path)} instead.
     *
     * @param filePath path to a .safetensors file
     * @return the cached or freshly-loaded result
     * @throws ProviderException on loading failure
     */
    public SafetensorLoadResult loadCached(Path filePath) {
        try {
            return loadCache.getOrLoad(filePath.toAbsolutePath().normalize(), ffmLoader);
        } catch (SafetensorException e) {
            throw wrapProviderException("Failed to load SafeTensors file: " + filePath, e, true);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Reactive API (Mutiny)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Asynchronously load a tensor and apply a transform function to it.
     *
     * <p>
     * The load and transform happen on the Vert.x worker pool so the calling
     * I/O thread is never blocked. The session (and thus the tensor's backing
     * memory) is kept alive for the duration of the transform, then closed.
     *
     * @param modelPath  model file or directory
     * @param tensorName the tensor to load
     * @param transform  function applied to the tensor (runs on worker thread)
     * @param <R>        result type
     * @return a {@link Uni} that resolves to the transform result
     */
    public <R> Uni<R> loadTensorAsync(
            Path modelPath, String tensorName, TensorTransform<R> transform) {

        return Uni.createFrom().item(() -> {
            try (SafetensorShardSession session = shardLoader.open(modelPath)) {
                SafetensorTensor tensor = session.tensor(tensorName);
                return transform.apply(tensor);
            } catch (SafetensorException e) {
                throw wrapProviderException(
                        "Failed to load tensor '" + tensorName + "' from " + modelPath, e, true);
            } catch (Exception e) {
                throw new ProviderException(
                        "Unexpected error loading tensor '" + tensorName + "': " + e.getMessage(), e);
            }
        }).runSubscriptionOn(io.vertx.mutiny.core.Vertx.currentContext() != null
                ? java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()
                : java.util.concurrent.Executors.newSingleThreadExecutor());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inspection API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Inspect the tensor catalogue of a model without loading tensor data.
     *
     * @param modelPath model file or directory
     * @return ordered list of tensor metadata
     */
    public List<SafetensorTensorInfo> inspectTensors(Path modelPath) {
        try {
            return shardLoader.inspectTensors(modelPath);
        } catch (SafetensorException e) {
            throw wrapProviderException(
                    "Failed to inspect SafeTensors at " + modelPath, e, false);
        }
    }

    /**
     * Load only the header of a single-file SafeTensors checkpoint.
     *
     * @param filePath .safetensors file path
     * @return the parsed header
     */
    public SafetensorHeader loadHeader(Path filePath) {
        try {
            return ffmLoader.loadHeaderOnly(filePath);
        } catch (SafetensorException e) {
            throw wrapProviderException(
                    "Failed to load header from " + filePath, e, false);
        }
    }

    /**
     * Check whether a path is a valid SafeTensors model (file or directory).
     *
     * @param modelPath candidate path
     * @return {@code true} if the path looks like a usable SafeTensors model
     */
    public boolean isValidModel(Path modelPath) {
        if (modelPath == null || !Files.exists(modelPath))
            return false;
        if (Files.isRegularFile(modelPath)) {
            String name = modelPath.getFileName().toString().toLowerCase();
            return name.endsWith(".safetensors") || name.endsWith(".safetensor");
        }
        if (Files.isDirectory(modelPath)) {
            return SafetensorShardLoader.SafetensorShardSession.class != null
                    && (SafetensorShardIndex
                            .isShardedModel(modelPath)
                            || hasSafetensorFileInDir(modelPath));
        }
        return false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cache management
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Invalidate the cache entry for a specific file, forcing a reload on next
     * access.
     * Useful when a model file has been updated in-place.
     *
     * @param filePath the file to invalidate
     */
    public void invalidateCache(Path filePath) {
        loadCache.invalidate(filePath.toAbsolutePath().normalize());
    }

    /** Clear all cached load results and release associated off-heap memory. */
    public void clearCache() {
        loadCache.invalidateAll();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Functional interface
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * A transform applied to a tensor within a managed session.
     * The tensor passed to {@link #apply} must not be stored beyond the method
     * call.
     *
     * @param <R> result type
     */
    @FunctionalInterface
    public interface TensorTransform<R> {
        R apply(SafetensorTensor tensor) throws Exception;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static ProviderException wrapProviderException(
            String message, SafetensorException cause, boolean retryable) {
        return new ProviderException(message + " (retryable=" + retryable + ")", cause);
    }

    private static boolean hasSafetensorFileInDir(Path dir) {
        try (var stream = Files.list(dir)) {
            return stream.anyMatch(p -> {
                String name = p.getFileName().toString().toLowerCase();
                return name.endsWith(".safetensors") || name.endsWith(".safetensor");
            });
        } catch (Exception e) {
            return false;
        }
    }
}
