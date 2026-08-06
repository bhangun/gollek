/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelRouter.java
 * ─────────────────
 * A/B model routing, canary deploys, and shadow mode for safe model upgrades.
 *
 * Routing modes
 * ═════════════
 *
 * 1. SHADOW  — all traffic goes to the primary model; secondary runs in the
 *              background and logs side-by-side outputs for quality comparison.
 *              Clients see only the primary's response. Zero user impact.
 *
 * 2. CANARY  — configurable % of traffic routes to the secondary model.
 *              E.g. 5% → secondary, 95% → primary. Ramped up as confidence grows.
 *
 * 3. A_B     — 50/50 split (or any ratio). Both models serve real traffic.
 *              Quality metrics are compared via admin dashboard.
 *
 * 4. PASSTHROUGH — all traffic to primary (default, no secondary loaded).
 *
 * Promotion workflow
 * ══════════════════
 *   1. Load secondary model:  POST /v1/routing/models/secondary {path}
 *   2. Start shadow mode:     PUT /v1/routing/mode {mode:"shadow"}
 *   3. Monitor divergence:    GET /v1/routing/stats
 *   4. Canary 5%:             PUT /v1/routing/mode {mode:"canary", canaryPct:5}
 *   5. Ramp up to 50%:        PUT /v1/routing/mode {mode:"a_b"}
 *   6. Promote secondary:     PUT /v1/routing/promote
 *   7. Unload old primary:    PUT /v1/routing/mode {mode:"passthrough"}
 *
 * Configuration
 * ═════════════
 *   gollek.routing.mode=passthrough
 *   gollek.routing.canary-pct=5
 *   gollek.routing.shadow-async=true  (shadow runs in background, not on critical path)
 */
package tech.kayys.gollek.safetensor.engine.routing;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * A/B model router supporting canary deploys, shadow mode, and traffic
 * splitting.
 *
 * <p>
 * Inject instead of calling {@link DirectInferenceEngine} directly when
 * model routing is required.
 */
@ApplicationScoped
public class ModelRouter {

    private static final Logger log = Logger.getLogger(ModelRouter.class);

    public enum Mode {
        PASSTHROUGH, SHADOW, CANARY, A_B
    }

    @ConfigProperty(name = "gollek.routing.mode", defaultValue = "passthrough")
    String configMode;

    @ConfigProperty(name = "gollek.routing.canary-pct", defaultValue = "5")
    int defaultCanaryPct;

    @ConfigProperty(name = "gollek.routing.shadow-async", defaultValue = "true")
    boolean shadowAsync;

    @Inject
    DirectInferenceEngine engine;

    // ── State ─────────────────────────────────────────────────────────────────

    private volatile Mode mode = Mode.PASSTHROUGH;
    private volatile Path primary = null;
    private volatile Path secondary = null;
    private volatile int canaryPct = 5; // % of requests routed to secondary

    // Rolling window statistics
    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong primaryRequests = new AtomicLong(0);
    private final AtomicLong secondaryRequests = new AtomicLong(0);
    private final AtomicLong shadowDivergences = new AtomicLong(0);
    private final AtomicLong shadowComparisons = new AtomicLong(0);

    private final ExecutorService shadowPool = Executors.newVirtualThreadPerTaskExecutor();

    private final Random rng = new Random();

    // ─────────────────────────────────────────────────────────────────────────
    // Configuration API
    // ─────────────────────────────────────────────────────────────────────────

    public void setPrimary(Path modelPath) {
        this.primary = modelPath;
        engine.loadModel(modelPath);
        log.infof("ModelRouter: primary set to %s", modelPath.getFileName());
    }

    public void setSecondary(Path modelPath) {
        this.secondary = modelPath;
        engine.loadModel(modelPath);
        log.infof("ModelRouter: secondary set to %s", modelPath.getFileName());
    }

    public void setMode(Mode newMode, int canaryPercent) {
        this.mode = newMode;
        this.canaryPct = Math.max(0, Math.min(100, canaryPercent));
        log.infof("ModelRouter: mode=%s canaryPct=%d", newMode, this.canaryPct);
    }

    /**
     * Promote secondary to primary. Old primary is retained as the new secondary
     * (for rollback).
     */
    public void promote() {
        Path old = this.primary;
        this.primary = this.secondary;
        this.secondary = old;
        log.infof("ModelRouter: promoted secondary to primary [%s → %s]",
                old != null ? old.getFileName() : "none",
                primary != null ? primary.getFileName() : "none");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Routing
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Route a generation request according to the current mode.
     *
     * @param prompt formatted prompt
     * @param cfg    generation config
     * @return response from the routed model
     */
    public Uni<InferenceResponse> generate(String prompt, GenerationConfig cfg) {
        if (primary == null)
            return Uni.createFrom().failure(new IllegalStateException(
                    "ModelRouter: no primary model set. Call setPrimary() first."));

        totalRequests.incrementAndGet();

        return switch (mode) {
            case PASSTHROUGH -> routePrimary(prompt, cfg);
            case SHADOW -> routeShadow(prompt, cfg);
            case CANARY -> routeCanary(prompt, cfg);
            case A_B -> routeAB(prompt, cfg);
        };
    }

    private Uni<InferenceResponse> routePrimary(String prompt, GenerationConfig cfg) {
        primaryRequests.incrementAndGet();
        return engine.generate(prompt, primary, cfg);
    }

    private Uni<InferenceResponse> routeShadow(String prompt, GenerationConfig cfg) {
        primaryRequests.incrementAndGet();
        Uni<InferenceResponse> primaryResp = engine.generate(prompt, primary, cfg);

        if (secondary != null) {
            // Shadow: run secondary in background, never return its response
            if (shadowAsync) {
                shadowPool.submit(() -> {
                    try {
                        InferenceResponse sec = engine.generate(prompt, secondary, cfg)
                                .await().indefinitely();
                        secondaryRequests.incrementAndGet();
                        // Compare outputs (simple length/divergence metric)
                        InferenceResponse prim = engine.generate(prompt, primary, cfg)
                                .await().indefinitely();
                        shadowComparisons.incrementAndGet();
                        if (!outputsSimilar(prim.getContent(), sec.getContent())) {
                            shadowDivergences.incrementAndGet();
                            log.debugf("Shadow divergence: primary=%d chars secondary=%d chars",
                                    prim.getContent().length(), sec.getContent().length());
                        }
                    } catch (Exception e) {
                        log.tracef("Shadow secondary failed: %s", e.getMessage());
                    }
                });
            }
        }
        return primaryResp;
    }

    private Uni<InferenceResponse> routeCanary(String prompt, GenerationConfig cfg) {
        if (secondary != null && rng.nextInt(100) < canaryPct) {
            secondaryRequests.incrementAndGet();
            log.tracef("Canary: routing to secondary (%d%%)", canaryPct);
            return engine.generate(prompt, secondary, cfg);
        }
        primaryRequests.incrementAndGet();
        return engine.generate(prompt, primary, cfg);
    }

    private Uni<InferenceResponse> routeAB(String prompt, GenerationConfig cfg) {
        if (secondary != null && rng.nextBoolean()) {
            secondaryRequests.incrementAndGet();
            return engine.generate(prompt, secondary, cfg);
        }
        primaryRequests.incrementAndGet();
        return engine.generate(prompt, primary, cfg);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stats
    // ─────────────────────────────────────────────────────────────────────────

    public RoutingStats stats() {
        double shadowDivRate = shadowComparisons.get() > 0
                ? shadowDivergences.get() * 100.0 / shadowComparisons.get()
                : 0;
        return new RoutingStats(
                mode,
                primary != null ? primary.getFileName().toString() : null,
                secondary != null ? secondary.getFileName().toString() : null,
                canaryPct,
                totalRequests.get(),
                primaryRequests.get(),
                secondaryRequests.get(),
                shadowDivRate);
    }

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Simple output similarity check based on token-level Jaccard similarity.
     * Returns true when similarity ≥ 0.7.
     */
    private static boolean outputsSimilar(String a, String b) {
        if (a == null || b == null)
            return false;
        Set<String> tokA = new HashSet<>(Arrays.asList(a.split("\\s+")));
        Set<String> tokB = new HashSet<>(Arrays.asList(b.split("\\s+")));
        Set<String> union = new HashSet<>(tokA);
        union.addAll(tokB);
        if (union.isEmpty())
            return true;
        Set<String> inter = new HashSet<>(tokA);
        inter.retainAll(tokB);
        return (double) inter.size() / union.size() >= 0.7;
    }

    public record RoutingStats(
            Mode mode,
            String primaryModel,
            String secondaryModel,
            int canaryPct,
            long totalRequests,
            long primaryRequests,
            long secondaryRequests,
            double shadowDivergenceRate) {
    }
}
