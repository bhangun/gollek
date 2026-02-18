/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.gollek.plugin.observability;

/**
 * Interface for exporting inference metrics.
 * Implementations may export to Prometheus, OpenTelemetry, logs, or custom systems.
 */
public interface MetricsExporter {

    /**
     * Export an inference trace.
     *
     * @param trace the inference trace to export
     */
    void export(InferenceTrace trace);

    /**
     * Flush any buffered metrics.
     */
    default void flush() {
        // Default: no-op
    }

    /**
     * Shutdown the exporter and release resources.
     */
    default void shutdown() {
        flush();
    }
}
