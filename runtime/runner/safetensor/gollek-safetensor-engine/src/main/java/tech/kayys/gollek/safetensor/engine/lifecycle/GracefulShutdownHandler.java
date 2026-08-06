/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * GracefulShutdownHandler.java
 * ────────────────────────────
 * Handles graceful shutdown of in-flight inference requests.
 *
 * Configuration:
 *   gollek.shutdown.timeout-s=30
 *   gollek.shutdown.quiet-period-s=5
 */
package tech.kayys.gollek.safetensor.engine.lifecycle;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import io.quarkus.runtime.ShutdownEvent;
import io.quarkus.runtime.StartupEvent;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Graceful shutdown handler — waits for in-flight requests to complete.
 */
@ApplicationScoped
public class GracefulShutdownHandler {

    private static final Logger log = Logger.getLogger(GracefulShutdownHandler.class);

    @ConfigProperty(name = "gollek.shutdown.timeout-s", defaultValue = "30")
    int shutdownTimeoutS;

    @ConfigProperty(name = "gollek.shutdown.quiet-period-s", defaultValue = "5")
    int quietPeriodS;

    private final AtomicInteger activeRequests = new AtomicInteger(0);
    private final AtomicBoolean shuttingDown = new AtomicBoolean(false);
    private final CountDownLatch shutdownLatch = new CountDownLatch(1);

    void onStart(@Observes StartupEvent ev) {
        log.info("GracefulShutdownHandler initialized");
    }

    @jakarta.annotation.PreDestroy
    public void shutdown() {
        log.infof("Initiating graceful shutdown (timeout=%ds, quiet=%ds)...",
                shutdownTimeoutS, quietPeriodS);
        shuttingDown.set(true);

        try {
            // Wait for active requests to complete
            if (activeRequests.get() > 0) {
                log.infof("Waiting for %d active requests to complete...", activeRequests.get());
                boolean completed = shutdownLatch.await(shutdownTimeoutS, TimeUnit.SECONDS);
                if (!completed) {
                    log.warnf("Shutdown timeout exceeded — %d requests still active", activeRequests.get());
                } else {
                    log.info("All active requests completed");
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.warn("Shutdown interrupted");
        }

        log.info("Graceful shutdown complete");
    }

    /**
     * Register entry into an inference request.
     * 
     * @throws IllegalStateException if shutdown is in progress
     */
    public void enterRequest() {
        if (shuttingDown.get()) {
            throw new IllegalStateException("Shutdown in progress — rejecting new requests");
        }
        activeRequests.incrementAndGet();
    }

    /**
     * Register exit from an inference request.
     */
    public void exitRequest() {
        int remaining = activeRequests.decrementAndGet();
        if (remaining == 0 && shuttingDown.get()) {
            shutdownLatch.countDown();
        }
    }

    /**
     * Check if shutdown is in progress.
     */
    public boolean isShuttingDown() {
        return shuttingDown.get();
    }

    /**
     * Get count of active requests.
     */
    public int getActiveRequests() {
        return activeRequests.get();
    }
}
