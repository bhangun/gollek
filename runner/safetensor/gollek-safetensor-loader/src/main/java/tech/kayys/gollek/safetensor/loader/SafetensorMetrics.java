/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorMetrics.java
 * ──────────────────────
 * Micrometer-based metrics recorder for SafeTensors loading operations.
 *
 * Metrics emitted
 * ═══════════════
 *   gollek.safetensor.load.duration   (Timer)    — load wall-clock time
 *   gollek.safetensor.load.bytes      (Counter)  — bytes loaded (cumulative)
 *   gollek.safetensor.load.total      (Counter)  — total load operations
 *   gollek.safetensor.load.errors     (Counter)  — load failures
 *   gollek.safetensor.open.files      (Gauge)    — currently open mmap'd files
 *
 * Tags on every metric
 * ════════════════════
 *   mode   = MMAP | COPY
 *   status = success | failure
 *
 * The class is ApplicationScoped so it holds the counters for the lifetime of
 * the application.  The Gauge for open files uses an AtomicLong that is
 * incremented / decremented by the loader lifecycle.
 */
package tech.kayys.gollek.safetensor.loader;

import io.micrometer.core.instrument.*;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Micrometer metrics for SafeTensors loader.
 *
 * <p>
 * Inject this bean wherever you need to record load / error events.
 * The metrics are automatically published to whatever Micrometer registry
 * is configured in the Quarkus application (Prometheus, Datadog, etc.).
 */
@ApplicationScoped
public class SafetensorMetrics {

        // ── Metric name constants ─────────────────────────────────────────────────

        private static final String PREFIX = "gollek.safetensor";
        private static final String LOAD_DURATION = PREFIX + ".load.duration";
        private static final String LOAD_BYTES = PREFIX + ".load.bytes";
        private static final String LOAD_TOTAL = PREFIX + ".load.total";
        private static final String LOAD_ERRORS = PREFIX + ".load.errors";
        private static final String OPEN_FILES = PREFIX + ".open.files";

        // ── Tag key constants ─────────────────────────────────────────────────────

        private static final String TAG_MODE = "mode";
        private static final String TAG_STATUS = "status";

        // ─────────────────────────────────────────────────────────────────────────

        @Inject
        MeterRegistry registry;

        /** Gauge backing — tracks currently open mmap'd / native segments. */
        private final AtomicLong openFilesGauge = new AtomicLong(0);

        // ─────────────────────────────────────────────────────────────────────────

        @PostConstruct
        void registerGauge() {
                Gauge.builder(OPEN_FILES, openFilesGauge, AtomicLong::doubleValue)
                                .description("Number of currently open (mmap'd or native) SafeTensors files")
                                .register(registry);
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Public API
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * Record a successful load operation.
         *
         * @param path      the loaded file path (used for filename tag)
         * @param mode      MMAP or COPY
         * @param byteCount total bytes in the tensor data region
         * @param elapsed   wall-clock time of the load call
         */
        public void recordLoad(
                        Path path,
                        SafetensorLoadResult.LoadMode mode,
                        long byteCount,
                        Duration elapsed) {

                String modeTag = mode.name();

                // Duration timer
                Timer.builder(LOAD_DURATION)
                                .description("Time to load a SafeTensors file")
                                .tag(TAG_MODE, modeTag)
                                .tag(TAG_STATUS, "success")
                                .register(registry)
                                .record(elapsed);

                // Byte counter
                Counter.builder(LOAD_BYTES)
                                .description("Bytes loaded from SafeTensors files")
                                .tag(TAG_MODE, modeTag)
                                .register(registry)
                                .increment(byteCount);

                // Total operations counter
                Counter.builder(LOAD_TOTAL)
                                .description("Total SafeTensors load operations")
                                .tag(TAG_MODE, modeTag)
                                .tag(TAG_STATUS, "success")
                                .register(registry)
                                .increment();

                // Track open file
                openFilesGauge.incrementAndGet();
        }

        /**
         * Record a failed load attempt.
         *
         * @param path  the file that failed to load
         * @param cause the exception class name (for the error-type tag)
         */
        public void recordLoadError(Path path, String cause) {
                Counter.builder(LOAD_ERRORS)
                                .description("SafeTensors load failures")
                                .tag("error_type", cause != null ? cause : "unknown")
                                .tag(TAG_STATUS, "failure")
                                .register(registry)
                                .increment();

                Counter.builder(LOAD_TOTAL)
                                .description("Total SafeTensors load operations")
                                .tag(TAG_MODE, "unknown")
                                .tag(TAG_STATUS, "failure")
                                .register(registry)
                                .increment();
        }

        /**
         * Decrement the open-files gauge when a {@link SafetensorLoadResult} is closed.
         * Call this from the load result's close hook.
         */
        public void recordClose() {
                openFilesGauge.decrementAndGet();
        }

        /**
         * Returns the current count of open files. Useful for health checks.
         */
        public long openFileCount() {
                return openFilesGauge.get();
        }
}
