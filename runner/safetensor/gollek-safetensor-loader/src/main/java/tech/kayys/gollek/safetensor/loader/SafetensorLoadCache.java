/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorLoadCache.java
 * ────────────────────────
 * Thread-safe, bounded LRU cache for {@link SafetensorLoadResult} objects.
 *
 * Motivation
 * ══════════
 * Memory-mapping a large SafeTensors file is an O(1) operation (mmap is lazy),
 * but the syscall itself, plus header parsing, still adds measurable latency
 * on the first request to each model.  A cache avoids this overhead on
 * subsequent requests to the same model file in a multi-tenant server.
 *
 * Eviction policy
 * ═══════════════
 * LRU with a configurable maximum size and idle TTL.
 * When an entry is evicted the underlying {@link SafetensorLoadResult} is
 * closed, which unmaps the file from the process address space.
 *
 * Thread-safety
 * ═════════════
 * All methods are synchronized on the internal LinkedHashMap to ensure
 * correct LRU ordering.  Lock contention is minimal because load operations
 * are long compared to the cache lock hold time.
 */
package tech.kayys.gollek.safetensor.loader;

import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.jboss.logging.Logger;

import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Bounded LRU cache for {@link SafetensorLoadResult} objects.
 *
 * <p>
 * Use {@link #getOrLoad(Path, SafetensorFFMLoader)} to load a file with
 * automatic cache lookup and insertion.
 */
@ApplicationScoped
public class SafetensorLoadCache {

    private static final Logger log = Logger.getLogger(SafetensorLoadCache.class);

    @Inject
    SafetensorLoaderConfig config;

    @Inject
    SafetensorMetrics metrics;

    /** Underlying LRU map. Access is synchronized on {@code this}. */
    private LinkedHashMap<Path, CacheEntry> cache;

    // ─────────────────────────────────────────────────────────────────────────

    private void ensureInitialized() {
        if (cache == null) {
            int maxSize = config.cache().maxSize();
            // Access-ordered LinkedHashMap with automatic eldest-entry removal
            cache = new LinkedHashMap<>(maxSize * 2, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Path, CacheEntry> eldest) {
                    boolean evict = size() > maxSize;
                    if (evict) {
                        log.debugf("SafetensorLoadCache: evicting [%s] (LRU, size limit=%d)",
                                eldest.getKey().getFileName(), maxSize);
                        closeQuietly(eldest.getValue().result());
                    }
                    return evict;
                }
            };
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Return a cached {@link SafetensorLoadResult} for the given path, or load
     * and cache it via the provided loader.
     *
     * @param path   absolute, normalized path to the .safetensors file
     * @param loader loader to use on cache miss
     * @return a live (non-closed) load result
     */
    public synchronized SafetensorLoadResult getOrLoad(Path path, SafetensorFFMLoader loader) {
        if (!config.cache().enabled()) {
            // Cache disabled — every call produces a new result
            return loader.load(path);
        }

        ensureInitialized();

        CacheEntry entry = cache.get(path);
        if (entry != null && !entry.result().isClosed() && !isExpired(entry)) {
            log.debugf("SafetensorLoadCache: HIT [%s]", path.getFileName());
            return entry.result();
        }

        // Cache miss or stale/closed entry
        if (entry != null) {
            log.debugf("SafetensorLoadCache: STALE/CLOSED [%s], reloading", path.getFileName());
            closeQuietly(entry.result());
        }

        log.debugf("SafetensorLoadCache: MISS [%s], loading", path.getFileName());
        SafetensorLoadResult result = loader.load(path);
        cache.put(path, new CacheEntry(result, Instant.now()));
        return result;
    }

    /**
     * Explicitly invalidate a cache entry (e.g. when a model file has been
     * updated).
     *
     * @param path path to invalidate
     */
    public synchronized void invalidate(Path path) {
        if (cache == null)
            return;
        CacheEntry entry = cache.remove(path);
        if (entry != null) {
            log.infof("SafetensorLoadCache: invalidated [%s]", path.getFileName());
            closeQuietly(entry.result());
        }
    }

    /** Invalidate all cached entries and release their memory. */
    public synchronized void invalidateAll() {
        if (cache == null)
            return;
        log.infof("SafetensorLoadCache: invalidating all %d entries", cache.size());
        cache.values().forEach(e -> closeQuietly(e.result()));
        cache.clear();
    }

    /** Current number of cached entries. */
    public synchronized int size() {
        return cache != null ? cache.size() : 0;
    }

    // ─────────────────────────────────────────────────────────────────────────

    @PreDestroy
    void onShutdown() {
        log.info("SafetensorLoadCache: application shutdown — releasing all mmap'd files");
        invalidateAll();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal types
    // ─────────────────────────────────────────────────────────────────────────

    private boolean isExpired(CacheEntry entry) {
        long ttlSeconds = config.cache().ttlSeconds();
        if (ttlSeconds <= 0)
            return false;
        return Instant.now().isAfter(entry.insertedAt().plusSeconds(ttlSeconds));
    }

    private void closeQuietly(SafetensorLoadResult result) {
        try {
            if (result != null && !result.isClosed()) {
                result.close();
                metrics.recordClose();
            }
        } catch (Exception e) {
            log.warnf(e, "SafetensorLoadCache: error closing evicted entry");
        }
    }

    /** Internal cache entry: result + insertion timestamp for TTL. */
    private record CacheEntry(SafetensorLoadResult result, Instant insertedAt) {
    }
}
