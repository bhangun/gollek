/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */
package tech.kayys.gollek.plugin.observability;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.exception.PluginException;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * Resilience plugin that enforces circuit-breaker and bulkhead patterns at the
 * {@link InferencePhase#PROVIDER_DISPATCH} phase.
 *
 * <h3>Circuit Breaker</h3>
 * Per-provider state machine with three states:
 * <ul>
 *   <li><b>CLOSED</b>: normal operation.</li>
 *   <li><b>OPEN</b>: short-circuits all requests after {@code failureThreshold}
 *       consecutive failures, entering a recovery window of
 *       {@code recoveryWindowMs} ms.</li>
 *   <li><b>HALF_OPEN</b>: one probe request allowed; success transitions back
 *       to CLOSED, failure re-opens.</li>
 * </ul>
 *
 * <h3>Bulkhead</h3>
 * Per-tenant concurrency limit via {@link Semaphore}.  Requests that cannot
 * immediately acquire a permit are rejected with HTTP 429-equivalent error.
 *
 * <p>The selected provider is stored in the execution context variable
 * {@code "selectedProvider"} by the routing phase; this plugin reads it.
 */
@ApplicationScoped
public class ReliabilityPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(ReliabilityPlugin.class.getName());

    /** Number of consecutive failures before opening the circuit. */
    private static final int FAILURE_THRESHOLD = 5;

    /** Time the circuit stays OPEN before transitioning to HALF_OPEN. */
    private static final long RECOVERY_WINDOW_MS = 30_000L;

    /** Maximum concurrent requests per tenant. */
    private static final int BULKHEAD_MAX_CONCURRENT = 20;

    // Circuit breaker state per provider
    private final Map<String, CircuitState> circuitStates    = new ConcurrentHashMap<>();
    private final Map<String, AtomicInteger> failureCounts   = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong>    openSinceMs     = new ConcurrentHashMap<>();

    // Bulkhead: semaphore per tenantId
    private final Map<String, Semaphore> tenantSemaphores = new ConcurrentHashMap<>();

    @Override
    public String pluginId() {
        return "gollek.reliability";
    }

    @Override
    public String displayName() {
        return "Reliability Plugin (Circuit Breaker + Bulkhead)";
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.PROVIDER_DISPATCH;
    }

    @Override
    public int order() {
        return 1; // Must run first in PROVIDER_DISPATCH
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        String providerId = context.getVariable("selectedProvider", String.class).orElse("unknown");
        String tenantId   = resolveTenantId(context);

        // --- Circuit Breaker check ---
        CircuitState state = getCircuitState(providerId);
        switch (state) {
            case OPEN -> {
                long openSince = openSinceMs.computeIfAbsent(providerId, k -> new AtomicLong(0)).get();
                long elapsed   = Instant.now().toEpochMilli() - openSince;
                if (elapsed > RECOVERY_WINDOW_MS) {
                    // Transition to HALF_OPEN – allow one probe
                    circuitStates.put(providerId, CircuitState.HALF_OPEN);
                    LOG.info("[Circuit Breaker] HALF_OPEN for provider=" + providerId);
                } else {
                    throw new PluginException(
                        "Circuit OPEN for provider=" + providerId + ". Retry after " +
                        ((RECOVERY_WINDOW_MS - elapsed) / 1000) + "s",
                        pluginId());
                }
            }
            case HALF_OPEN -> LOG.fine("[Circuit Breaker] Probe request for provider=" + providerId);
            case CLOSED    -> LOG.finest("[Circuit Breaker] CLOSED – normal flow provider=" + providerId);
        }

        // --- Bulkhead check ---
        Semaphore semaphore = tenantSemaphores.computeIfAbsent(
            tenantId, k -> new Semaphore(BULKHEAD_MAX_CONCURRENT, true));

        boolean acquired = semaphore.tryAcquire();
        if (!acquired) {
            throw new PluginException(
                "Bulkhead capacity exhausted for tenant=" + tenantId + ". Request rejected.",
                pluginId());
        }

        // Store semaphore reference so cleanup phase can release it
        context.putVariable("_reliabilityBulkheadSemaphore", semaphore);
        context.putVariable("_reliabilityProvider", providerId);

        LOG.fine(() -> String.format(
            "[Reliability] Admitted | provider=%s tenant=%s permits=%d",
            providerId, tenantId, semaphore.availablePermits()));
    }

    // -----------------------------------------------------------------------
    // Called externally after PROVIDER_DISPATCH finishes (success / failure)
    // -----------------------------------------------------------------------

    /**
     * Records a successful provider call, resetting failure count and
     * closing the circuit if it was HALF_OPEN.
     */
    public void onProviderSuccess(ExecutionContext context) {
        String providerId = context.getVariable("_reliabilityProvider", String.class)
                                   .orElse("unknown");
        failureCounts.computeIfAbsent(providerId, k -> new AtomicInteger(0)).set(0);
        CircuitState current = getCircuitState(providerId);
        if (current == CircuitState.HALF_OPEN) {
            circuitStates.put(providerId, CircuitState.CLOSED);
            LOG.info("[Circuit Breaker] CLOSED (recovered) for provider=" + providerId);
        }
        releaseBulkhead(context);
    }

    /**
     * Records a failed provider call, possibly opening the circuit.
     */
    public void onProviderFailure(ExecutionContext context) {
        String providerId = context.getVariable("_reliabilityProvider", String.class)
                                   .orElse("unknown");
        AtomicInteger failures = failureCounts.computeIfAbsent(providerId, k -> new AtomicInteger(0));
        int count = failures.incrementAndGet();

        CircuitState current = getCircuitState(providerId);
        if (current == CircuitState.HALF_OPEN || count >= FAILURE_THRESHOLD) {
            circuitStates.put(providerId, CircuitState.OPEN);
            openSinceMs.computeIfAbsent(providerId, k -> new AtomicLong(0))
                       .set(Instant.now().toEpochMilli());
            LOG.warning("[Circuit Breaker] OPENED for provider=" + providerId +
                        " after " + count + " failures");
        }
        releaseBulkhead(context);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private CircuitState getCircuitState(String providerId) {
        return circuitStates.getOrDefault(providerId, CircuitState.CLOSED);
    }

    private String resolveTenantId(ExecutionContext context) {
        return context.getVariable("tenantId", String.class)
                      .orElseGet(() -> {
                          try { return context.requestContext().tenantId(); }
                          catch (Exception e) { return "default"; }
                      });
    }

    private void releaseBulkhead(ExecutionContext context) {
        context.getVariable("_reliabilityBulkheadSemaphore", Semaphore.class)
               .ifPresent(Semaphore::release);
    }

    /**
     * Simple three-state circuit breaker enum.
     */
    private enum CircuitState {
        CLOSED, OPEN, HALF_OPEN
    }
}
