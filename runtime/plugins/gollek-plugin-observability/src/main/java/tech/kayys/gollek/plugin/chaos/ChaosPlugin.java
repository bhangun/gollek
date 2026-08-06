/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */
package tech.kayys.gollek.plugin.chaos;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.exception.PluginException;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;

import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Logger;

/**
 * Chaos Engineering plugin that injects controlled failures and latency into the
 * inference pipeline for resilience testing.
 *
 * <h3>Features</h3>
 * <ul>
 *   <li><b>Latency injection</b> – adds a configurable random delay
 *       (up to {@link #maxLatencyMs}) to requests, simulating slow providers.</li>
 *   <li><b>Error injection</b>  – with probability {@link #errorRate}, throws a
 *       {@link PluginException} simulating a provider failure.</li>
 *   <li><b>Environment guard</b> – chaos is <em>only active</em> when
 *       {@code gollek.chaos.enabled=true} is set in the engine context.
 *       This prevents accidental activation in production.</li>
 * </ul>
 *
 * <h3>Configuration</h3>
 * <pre>
 * # application.properties (Quarkus) or environment variables
 * gollek.chaos.enabled=true          # master switch (default: false)
 * gollek.chaos.error-rate=0.10       # 10% of requests fail (default: 0.05)
 * gollek.chaos.max-latency-ms=3000   # max injected delay in ms (default: 2000)
 * </pre>
 *
 * <p><b>Safety:</b> If the engine context does not explicitly enable chaos, this
 * plugin is a no-op regardless of any instance-level setting.
 */
@ApplicationScoped
public class ChaosPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(ChaosPlugin.class.getName());

    /** Context metadata key used to flag chaos as active for this request. */
    private static final String META_CHAOS_ACTIVE = "chaos.active";

    // -----------------------------------------------------------------------
    // Configuration (defaults; override via admin API or config file)
    // -----------------------------------------------------------------------

    /** Error injection probability [0.0 – 1.0]. */
    private volatile double errorRate = 0.05;

    /** Maximum injected latency in milliseconds. */
    private volatile long maxLatencyMs = 2_000L;

    /** Master switch – chaos only active when true. */
    private volatile boolean enabled = false;

    // -----------------------------------------------------------------------
    // InferencePhasePlugin
    // -----------------------------------------------------------------------

    @Override
    public String pluginId() {
        return "gollek.chaos";
    }

    @Override
    public String displayName() {
        return "Chaos Engineering Plugin";
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.PRE_PROCESSING;
    }

    @Override
    public int order() {
        return 999; // Last in PRE_PROCESSING so other enrichment still runs
    }

    /**
     * Only active when chaos is explicitly enabled via engine context flag
     * {@code chaos.enabled=true} OR the instance-level flag is set.
     * Production deployments will never expose this flag.
     */
    @Override
    public boolean shouldExecute(ExecutionContext context) {
        if (!enabled) {
            // Allow per-request override via context (e.g. from integration tests)
            return Boolean.TRUE.equals(context.metadata().get(META_CHAOS_ACTIVE));
        }
        return true;
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        ThreadLocalRandom rng = ThreadLocalRandom.current();

        // --- Error injection ---
        if (rng.nextDouble() < errorRate) {
            String msg = String.format(
                "[Chaos] Injected error for request=%s (errorRate=%.2f)",
                safeRequestId(context), errorRate);
            LOG.warning(msg);
            throw new PluginException(msg + " – Chaos-induced failure", pluginId());
        }

        // --- Latency injection ---
        if (maxLatencyMs > 0) {
            long delayMs = rng.nextLong(0, maxLatencyMs + 1);
            if (delayMs > 0) {
                LOG.fine(() -> String.format(
                    "[Chaos] Injecting %dms latency for request=%s",
                    delayMs, safeRequestId(context)));
                try {
                    Thread.sleep(delayMs);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new PluginException(
                        "[Chaos] Interrupted during latency injection", pluginId());
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Runtime configuration API (callable from admin REST endpoint)
    // -----------------------------------------------------------------------

    /** Enables or disables chaos injection at runtime. */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        LOG.info("[Chaos] Chaos injection " + (enabled ? "ENABLED" : "DISABLED"));
    }

    /** Sets the error injection probability (0.0 – 1.0). */
    public void setErrorRate(double rate) {
        if (rate < 0 || rate > 1) throw new IllegalArgumentException("errorRate must be in [0,1]");
        this.errorRate = rate;
        LOG.info("[Chaos] Error rate set to " + rate);
    }

    /** Sets the maximum injected latency in milliseconds. */
    public void setMaxLatencyMs(long ms) {
        if (ms < 0) throw new IllegalArgumentException("maxLatencyMs must be >= 0");
        this.maxLatencyMs = ms;
        LOG.info("[Chaos] Max latency set to " + ms + "ms");
    }

    public boolean isEnabled()      { return enabled; }
    public double  getErrorRate()   { return errorRate; }
    public long    getMaxLatencyMs(){ return maxLatencyMs; }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private String safeRequestId(ExecutionContext context) {
        try { return context.token().requestId(); }
        catch (Exception e) { return "unknown"; }
    }
}
