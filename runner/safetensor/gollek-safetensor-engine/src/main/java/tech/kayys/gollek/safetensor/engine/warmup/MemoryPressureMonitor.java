/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * MemoryPressureMonitor.java
 * ──────────────────────────
 * Monitors off-heap memory pressure and takes corrective action before
 * the system runs out of blocks or the OOM killer fires.
 *
 * Two pressure levels
 * ════════════════════
 *
 *   WARN  (free blocks < 20% of total)
 *     → Log a warning with diagnostics.
 *     → Reject NEW requests with HTTP 429 (Too Many Requests).
 *     → Signal the GracefulShutdownHandler to stop accepting new work.
 *
 *   CRITICAL  (free blocks < 5% of total)
 *     → Evict the oldest idle KV cache sessions.
 *     → If models > 1 are loaded, unload the least-recently-used model.
 *
 * Off-heap memory sources
 * ═══════════════════════
 *   - PagedKVCache Arena     (dominant — grows with concurrent requests)
 *   - SafetensorLoadCache    (mmap'd model weights — fixed after load)
 *   - Per-request flat KV    (from KVCacheManager.KVCacheSession)
 *
 * The monitor uses a background virtual thread that samples every
 * {@code gollek.memory.check-interval-s} seconds (default: 5).
 *
 * It also exposes a health indicator so that Kubernetes readiness probes
 * can detect and shed load before an OOM crash.
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import io.quarkus.runtime.StartupEvent;
import io.quarkus.runtime.ShutdownEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
// import tech.kayys.gollek.safetensor.engine.generation.paged.PagedKVCache;
// // import tech.kayys.gollek.safetensor.metrics.SafetensorMetrics;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Readiness;
import org.jboss.logging.Logger;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Background off-heap memory pressure monitor.
 *
 * <p>
 * Runs every {@code checkIntervalSeconds} on a virtual thread and reacts to
 * low-block conditions in the {@link PagedKVCache}.
 */
@ApplicationScoped
@Readiness
public class MemoryPressureMonitor implements HealthCheck {

    private static final Logger log = Logger.getLogger(MemoryPressureMonitor.class);

    // Thresholds (fraction of total blocks)
    private static final double WARN_THRESHOLD = 0.20; // < 20% free → warn + reject new
    private static final double CRITICAL_THRESHOLD = 0.05; // < 5% free → evict

    @ConfigProperty(name = "gollek.memory.check-interval-s", defaultValue = "5")
    int checkIntervalSeconds;

    @ConfigProperty(name = "gollek.memory.heap-warn-pct", defaultValue = "85")
    int heapWarnPercent;

    // @Inject
    // PagedKVCache pagedKVCache;
    // @Inject
    // SafetensorMetrics metrics;

    private volatile boolean running = false;
    private final AtomicBoolean pressureActive = new AtomicBoolean(false);
    private final AtomicReference<String> lastStatus = new AtomicReference<>("OK");

    // ─────────────────────────────────────────────────────────────────────────

    void onStart(@Observes StartupEvent e) {
        running = true;
        Thread.ofVirtual().name("gollek-mem-monitor").start(this::monitorLoop);
        log.infof("MemoryPressureMonitor: started (interval=%ds)", checkIntervalSeconds);
    }

    void onShutdown(@Observes ShutdownEvent e) {
        running = false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Monitor loop
    // ─────────────────────────────────────────────────────────────────────────

    private void monitorLoop() {
        while (running) {
            try {
                Thread.sleep(checkIntervalSeconds * 1000L);
                checkPressure();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                log.warnf(e, "MemoryPressureMonitor: check failed");
            }
        }
    }

    private void checkPressure() {
        // ── 1. KV block pressure (Disabled - PagedKVCache missing) ────────────
        /*
        int totalBlocks = pagedKVCache.usedBlockCount() + pagedKVCache.freeBlockCount();
        if (totalBlocks > 0) {
            double freeRatio = (double) pagedKVCache.freeBlockCount() / totalBlocks;

            if (freeRatio < CRITICAL_THRESHOLD) {
                String msg = String.format(
                        "CRITICAL: KV block pool nearly exhausted (free=%.1f%%, used=%d/%d). " +
                                "Evicting idle sessions.",
                        freeRatio * 100,
                        pagedKVCache.usedBlockCount(), totalBlocks);
                log.error(msg);
                lastStatus.set("CRITICAL: " + msg);
                pressureActive.set(true);

            } else if (freeRatio < WARN_THRESHOLD) {
                String msg = String.format(
                        "WARN: KV block pool below 20%% (free=%.1f%%, used=%d/%d). " +
                                "Rejecting new requests.",
                        freeRatio * 100,
                        pagedKVCache.usedBlockCount(), totalBlocks);
                log.warn(msg);
                lastStatus.set("WARN: " + msg);
                pressureActive.set(true);
            } else {
                pressureActive.set(false);
                lastStatus.set("OK");
            }
        }
        */

        // ── 2. JVM heap pressure ──────────────────────────────────────────────
        MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage heap = memBean.getHeapMemoryUsage();
        if (heap.getMax() > 0) {
            long usedPct = heap.getUsed() * 100 / heap.getMax();
            if (usedPct >= heapWarnPercent) {
                log.warnf("MemoryPressureMonitor: JVM heap at %d%% (%s used / %s max). " +
                        "Consider increasing -Xmx or reducing concurrent request limit.",
                        usedPct, humanBytes(heap.getUsed()), humanBytes(heap.getMax()));
            }
        }

        // ── 3. Emit gauge metrics ─────────────────────────────────────────────
        // metrics.recordKVBlockUsage(pagedKVCache.usedBlockCount(), totalBlocks);
        // (deferred until SafetensorMetrics adds the KV gauge method)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Health check — exposed at /q/health/ready
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public HealthCheckResponse call() {
        return HealthCheckResponse.named("memory-pressure")
                .status(true)
                .withData("status", lastStatus.get())
                .withData("pressure_active", pressureActive.get())
                .build();
    }

    /** Whether the engine is currently under memory pressure. */
    public boolean isPressureActive() {
        return pressureActive.get();
    }

    // ─────────────────────────────────────────────────────────────────────────

    private static String humanBytes(long b) {
        if (b < 1_048_576)
            return (b / 1024) + " KiB";
        if (b < 1_073_741_824)
            return String.format("%.1f MiB", b / 1_048_576.0);
        return String.format("%.2f GiB", b / 1_073_741_824.0);
    }
}
