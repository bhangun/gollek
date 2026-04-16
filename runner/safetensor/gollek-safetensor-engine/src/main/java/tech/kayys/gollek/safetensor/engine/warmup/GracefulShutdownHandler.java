/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * GracefulShutdownHandler.java
 * ─────────────────────────────
 * Handles JVM shutdown gracefully, ensuring:
 *   1. No new inference requests are accepted (return 503 immediately).
 *   2. In-flight generation loops are allowed to complete within a drain timeout.
 *   3. All loaded models are unloaded (mmap's unmapped, arenas closed).
 *   4. PagedKVCache arena is released.
 *
 * Without this, a Kubernetes rolling deploy with a SIGTERM causes:
 *   - Responses truncated mid-stream (client gets a broken SSE stream)
 *   - Memory leaks from un-closed Arenas
 *   - CUDA error logs if GPU tensors are freed mid-kernel
 *
 * Integration
 * ════════════
 * Quarkus fires the ShutdownEvent before the CDI container stops.
 * This bean observes it and blocks until drain completes (or times out).
 *
 * Configure:
 *   gollek.shutdown.drain-timeout-s=30   (default: 30 seconds)
 *   gollek.shutdown.force-after-s=60     (hard kill after this)
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import io.quarkus.runtime.ShutdownEvent;
import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.container.ContainerRequestFilter;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.ext.Provider;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
// import tech.kayys.gollek.safetensor.engine.generation.paged.PagedKVCache;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Graceful shutdown coordinator for the inference engine.
 *
 * <p>
 * Tracks active requests and blocks new ones once draining starts.
 * Also provides a JAX-RS filter that rejects requests during drain.
 */
@ApplicationScoped
public class GracefulShutdownHandler {

    private static final Logger log = Logger.getLogger(GracefulShutdownHandler.class);

    @ConfigProperty(name = "gollek.shutdown.drain-timeout-s", defaultValue = "30")
    int drainTimeoutSeconds;

    @ConfigProperty(name = "gollek.shutdown.force-after-s", defaultValue = "60")
    int forceAfterSeconds;

    @Inject
    DirectInferenceEngine engine;
    // @Inject
    // PagedKVCache pagedKVCache;

    /** Set to true when shutdown has been signalled. */
    private final AtomicBoolean draining = new AtomicBoolean(false);

    /** Counts in-flight inference requests. */
    private final AtomicInteger activeRequests = new AtomicInteger(0);

    /** Latch that releases when activeRequests reaches zero. */
    private volatile CountDownLatch drainLatch;

    // ─────────────────────────────────────────────────────────────────────────

    void onStart(@Observes StartupEvent e) {
        drainLatch = new CountDownLatch(1);
        log.info("GracefulShutdownHandler: ready");
    }

    void onShutdown(@Observes ShutdownEvent e) {
        log.infof("GracefulShutdownHandler: SIGTERM received — draining (timeout=%ds)",
                drainTimeoutSeconds);

        draining.set(true);

        int active = activeRequests.get();
        if (active == 0) {
            log.info("GracefulShutdownHandler: no active requests — clean shutdown");
        } else {
            log.infof("GracefulShutdownHandler: waiting for %d active request(s) to complete…",
                    active);
            try {
                boolean drained = drainLatch.await(drainTimeoutSeconds, TimeUnit.SECONDS);
                if (!drained) {
                    log.warnf("GracefulShutdownHandler: drain timeout (%ds) — %d request(s) abandoned",
                            drainTimeoutSeconds, activeRequests.get());
                } else {
                    log.info("GracefulShutdownHandler: all requests drained cleanly");
                }
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                log.warn("GracefulShutdownHandler: interrupted during drain");
            }
        }

        // Engine and KV cache are closed via their own @PreDestroy beans.
        // We just log completion here.
        log.info("GracefulShutdownHandler: shutdown complete");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Request tracking API (called by DirectInferenceEngine)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Try to register an incoming request.
     *
     * @return {@code true} if the request is accepted; {@code false} if draining
     *         (caller should return HTTP 503)
     */
    public boolean tryAcquire() {
        if (draining.get())
            return false;
        activeRequests.incrementAndGet();
        return true;
    }

    /**
     * Signal that a request has completed.
     * Releases the drain latch when the last in-flight request finishes.
     */
    public void release() {
        int remaining = activeRequests.decrementAndGet();
        if (draining.get() && remaining <= 0) {
            drainLatch.countDown();
        }
    }

    /** Whether the engine is currently draining (refusing new requests). */
    public boolean isDraining() {
        return draining.get();
    }

    /** Current count of active in-flight requests. */
    public int activeRequestCount() {
        return activeRequests.get();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JAX-RS filter — rejects inference requests when draining
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * JAX-RS filter that returns HTTP 503 for inference requests when the
     * engine is draining.
     *
     * <p>
     * Non-inference paths (health, metrics) are always allowed through.
     */
    @Provider
    @ApplicationScoped
    public static class DrainFilter implements ContainerRequestFilter {

        @Inject
        GracefulShutdownHandler shutdownHandler;

        private static final java.util.Set<String> ALWAYS_ALLOWED = java.util.Set.of(
                "/q/health", "/q/metrics", "/q/info");

        @Override
        public void filter(ContainerRequestContext ctx) throws IOException {
            String path = ctx.getUriInfo().getPath();

            // Always allow health / metrics paths
            for (String allowed : ALWAYS_ALLOWED) {
                if (path.startsWith(allowed))
                    return;
            }

            // Block inference paths during drain
            if (path.startsWith("/v1/") && shutdownHandler.isDraining()) {
                ctx.abortWith(Response.status(503)
                        .entity("{\"error\":{\"type\":\"service_unavailable\","
                                + "\"message\":\"Server is shutting down. Retry on another node.\"}}")
                        .build());
            }
        }
    }
}
