/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorLoaderConfig.java
 * ───────────────────────────
 * SmallRye config mapping for the FFM-based SafeTensors loader.
 *
 * All properties live under the {@code gollek.safetensor.loader} namespace.
 *
 * Example application.properties:
 * ─────────────────────────────────────────────────────────────────────────────
 *   gollek.safetensor.loader.prefer-mmap=true
 *   gollek.safetensor.loader.read-chunk-bytes=8388608
 *   gollek.safetensor.loader.validation.strict=true
 *   gollek.safetensor.loader.validation.check-overlaps=true
 *   gollek.safetensor.loader.cache.enabled=true
 *   gollek.safetensor.loader.cache.max-size=8
 * ─────────────────────────────────────────────────────────────────────────────
 */
package tech.kayys.gollek.safetensor.loader;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

/**
 * Configuration contract for
 * {@link tech.kayys.gollek.inference.safetensor.ffm.SafetensorFFMLoader}.
 *
 * <p>
 * Inject this interface wherever loader-level configuration is needed:
 * 
 * <pre>{@code
 * @Inject
 * SafetensorLoaderConfig config;
 * }</pre>
 */
@ConfigMapping(prefix = "gollek.safetensor.loader")
public interface SafetensorLoaderConfig {

    // ── Loading strategy ──────────────────────────────────────────────────────

    /**
     * Whether to prefer memory-mapped I/O ({@code true}) or read-copy
     * ({@code false}).
     *
     * <p>
     * MMAP is preferred for large models on local filesystems: it is lazy,
     * zero-copy, and lets the OS manage page eviction. Set to {@code false} for
     * network mounts (NFS, CIFS) where mmap may be unsupported or unreliable.
     *
     * <p>
     * The loader automatically falls back to COPY mode if mmap fails.
     */
    @WithDefault("true")
    @WithName("prefer-mmap")
    boolean preferMmap();

    /**
     * Read-chunk size in bytes used in COPY mode.
     *
     * <p>
     * The file is read in chunks of this size into a staging heap buffer,
     * then copied into the off-heap native allocation.
     * Default: 8 MiB.
     */
    @WithDefault("8388608")
    @WithName("read-chunk-bytes")
    int readChunkBytes();

    // ── Validation ────────────────────────────────────────────────────────────

    /**
     * Validation settings applied during header parsing.
     */
    Validation validation();

    interface Validation {
        /**
         * Enable strict header validation.
         * When {@code true}, additional checks are performed:
         * <ul>
         * <li>AccelTensor data_offsets must be non-overlapping.
         * <li>Declared byte count must match dtype × shape.
         * <li>All offsets must be within the file boundary.
         * </ul>
         */
        @WithDefault("true")
        boolean strict();

        /**
         * Warn (but not fail) when tensors have zero byte length.
         * Zero-length tensors are valid per the spec but unusual.
         */
        @WithDefault("true")
        @WithName("warn-on-empty-tensors")
        boolean warnOnEmptyTensors();

        /**
         * Maximum allowed header JSON size in bytes.
         * Default: 104857600 (100 MiB). Increase only if you have models
         * with an extraordinarily large number of tensors.
         */
        @WithDefault("104857600")
        @WithName("max-header-bytes")
        long maxHeaderBytes();
    }

    // ── Load-result cache ─────────────────────────────────────────────────────

    /**
     * Cache settings for loaded SafeTensors files.
     *
     * <p>
     * When enabled, the loader keeps recently-used
     * {@link tech.kayys.gollek.inference.safetensor.ffm.SafetensorLoadResult}
     * objects alive in a bounded LRU cache to avoid repeated mmap calls for
     * frequently-accessed model checkpoints.
     */
    Cache cache();

    interface Cache {
        /**
         * Whether the load-result cache is enabled.
         * Default: {@code true}.
         */
        @WithDefault("true")
        boolean enabled();

        /**
         * Maximum number of simultaneously cached load results.
         * Each result keeps a file mmap'd (or a native copy) alive.
         * Default: 8 (suitable for a multi-tenant server loading multiple models).
         */
        @WithDefault("8")
        @WithName("max-size")
        int maxSize();

        /**
         * TTL (time-to-live) in seconds for idle cache entries.
         * A result that has not been accessed for this many seconds is evicted.
         * Default: 300 (5 minutes).
         */
        @WithDefault("300")
        @WithName("ttl-seconds")
        long ttlSeconds();
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /**
     * Whether to log a summary of each loaded file (tensor count, byte size,
     * load mode, elapsed time).
     * Default: {@code true}.
     */
    @WithDefault("true")
    @WithName("log-load-summary")
    boolean logLoadSummary();
}
