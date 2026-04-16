/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * MemoryPressureMonitor.java
 * ──────────────────────────
 * Monitors GPU/CPU memory pressure to prevent OOM during model operations.
 *
 * Configuration:
 *   gollek.memory.pressure.threshold=0.85  # 85% usage triggers pressure
 *   gollek.memory.check.interval-s=5
 */
package tech.kayys.gollek.safetensor.engine.lifecycle;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Memory pressure monitor — prevents OOM during hot-swap and model loading.
 */
@ApplicationScoped
public class MemoryPressureMonitor {

    private static final Logger log = Logger.getLogger(MemoryPressureMonitor.class);

    @ConfigProperty(name = "gollek.memory.pressure.threshold", defaultValue = "0.85")
    double pressureThreshold;

    @ConfigProperty(name = "gollek.memory.check.interval-s", defaultValue = "5")
    int checkIntervalS;

    private final MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
    private final AtomicBoolean pressureActive = new AtomicBoolean(false);
    private final ScheduledExecutorService scheduler;

    public MemoryPressureMonitor() {
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "memory-pressure-monitor");
            t.setDaemon(true);
            return t;
        });
        scheduler.scheduleAtFixedRate(this::checkMemory, 0, checkIntervalS, TimeUnit.SECONDS);
    }

    private void checkMemory() {
        MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
        double usageRatio = (double) heapUsage.getUsed() / heapUsage.getMax();

        boolean pressure = usageRatio > pressureThreshold;
        boolean changed = pressureActive.getAndSet(pressure);

        if (pressure && !changed) {
            log.warnf("Memory pressure ACTIVE: heap usage %.1f%% (threshold: %.1f%%)",
                    usageRatio * 100, pressureThreshold * 100);
        } else if (!pressure && changed) {
            log.infof("Memory pressure CLEARED: heap usage %.1f%%", usageRatio * 100);
        }
    }

    /**
     * Check if memory pressure is currently active.
     * 
     * @return true if memory usage exceeds threshold
     */
    public boolean isPressureActive() {
        return pressureActive.get();
    }

    /**
     * Get current memory usage ratio.
     * 
     * @return heap usage as ratio (0.0 to 1.0)
     */
    public double getUsageRatio() {
        MemoryUsage usage = memoryBean.getHeapMemoryUsage();
        return (double) usage.getUsed() / usage.getMax();
    }

    /**
     * Check if there's enough free memory for a new model.
     * 
     * @param requiredBytes estimated memory requirement
     * @return true if sufficient memory is available
     */
    public boolean hasFreeMemory(long requiredBytes) {
        MemoryUsage usage = memoryBean.getHeapMemoryUsage();
        long available = usage.getMax() - usage.getUsed();
        return available >= requiredBytes;
    }
}
