/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * MultiTenantQuotaService.java
 * ─────────────────────────────
 * Per-tenant token budget enforcement and cost tracking.
 *
 * Design
 * ══════
 * Each tenant has:
 *   - A token budget (input + output tokens combined)
 *   - A usage counter that resets at the billing period boundary
 *   - A cost accumulator (tokens × price)
 *   - A hard limit that triggers HTTP 402 when exceeded
 *   - A soft warning threshold (90% of budget)
 *
 * Storage backends
 * ════════════════
 * Default: in-memory ConcurrentHashMap (single-node, resets on restart).
 * Production: replace with Redis INCR + TTL via Quarkus Redis client
 * by injecting io.quarkus.redis.datasource.ReactiveRedisDataSource.
 *
 * The interface is kept storage-agnostic so the backend can be swapped
 * without changing callers.
 *
 * Tenant resolution
 * ════════════════
 * Tenant ID comes from ProviderRequest in this priority order:
 *   metadata["tenantId"] → userId → apiKey → "community"
 *
 * Per-tenant limits are configured via:
 *   gollek.quota.tenant.<tenantId>.monthly-tokens=10000000
 *   gollek.quota.tenant.<tenantId>.cost-per-1k-input=0.001
 *   gollek.quota.tenant.<tenantId>.cost-per-1k-output=0.002
 *   gollek.quota.default.monthly-tokens=1000000  (community tier)
 */
package tech.kayys.gollek.safetensor.engine.tooling;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.Config;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * In-memory multi-tenant quota service with billing-period reset.
 *
 * <p>
 * Inject and call {@link #tryConsume} before each inference request.
 * Call {@link #recordUsage} after completion to debit the token count.
 */
@ApplicationScoped
public class MultiTenantQuotaService {

    private static final Logger log = Logger.getLogger(MultiTenantQuotaService.class);

    @ConfigProperty(name = "gollek.quota.enabled", defaultValue = "true")
    boolean enabled;

    @ConfigProperty(name = "gollek.quota.default.monthly-tokens", defaultValue = "1000000")
    long defaultMonthlyTokens;

    @ConfigProperty(name = "gollek.quota.default.cost-per-1k-input", defaultValue = "0.0")
    double defaultCostPer1kInput;

    @ConfigProperty(name = "gollek.quota.default.cost-per-1k-output", defaultValue = "0.0")
    double defaultCostPer1kOutput;

    @Inject
    Config mpConfig;

    /** tenantId → current period usage state. */
    private final ConcurrentHashMap<String, TenantState> tenants = new ConcurrentHashMap<>();

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Check whether a tenant has remaining quota.
     *
     * @param tenantId        resolved tenant identifier
     * @param estimatedTokens rough estimate of tokens this request will use
     * @return {@code true} if the request should proceed, {@code false} if quota is
     *         exhausted
     */
    public boolean tryConsume(String tenantId, int estimatedTokens) {
        if (!enabled)
            return true;
        TenantState state = getOrCreate(tenantId);
        state.resetIfNewPeriod();
        long budget = getBudget(tenantId);
        long current = state.usedTokens.get();
        if (current >= budget) {
            log.warnf("Quota exhausted for tenant '%s': %d/%d tokens used this period",
                    tenantId, current, budget);
            return false;
        }
        if (current + estimatedTokens > budget * 0.9) {
            log.infof("Quota warning for tenant '%s': %.0f%% used",
                    tenantId, current * 100.0 / budget);
        }
        return true;
    }

    /**
     * Record actual token usage after a request completes.
     *
     * @param tenantId     tenant identifier
     * @param inputTokens  prompt token count
     * @param outputTokens generated token count
     */
    public void recordUsage(String tenantId, int inputTokens, int outputTokens) {
        if (!enabled)
            return;
        TenantState state = getOrCreate(tenantId);
        state.resetIfNewPeriod();
        int total = inputTokens + outputTokens;
        state.usedTokens.addAndGet(total);
        state.usedInputTokens.addAndGet(inputTokens);
        state.usedOutputTokens.addAndGet(outputTokens);

        // Accumulate cost
        double costIn = (inputTokens / 1000.0) * getCostPer1kInput(tenantId);
        double costOut = (outputTokens / 1000.0) * getCostPer1kOutput(tenantId);
        state.accumulatedCostMicros.addAndGet((long) ((costIn + costOut) * 1_000_000));

        log.tracef("Quota recorded: tenant=%s input=%d output=%d total=%d used=%d",
                tenantId, inputTokens, outputTokens, total,
                state.usedTokens.get());
    }

    /**
     * Get current usage statistics for a tenant.
     */
    public QuotaStats stats(String tenantId) {
        TenantState state = getOrCreate(tenantId);
        long budget = getBudget(tenantId);
        return new QuotaStats(
                tenantId,
                state.usedTokens.get(),
                state.usedInputTokens.get(),
                state.usedOutputTokens.get(),
                budget,
                state.accumulatedCostMicros.get() / 1_000_000.0,
                state.periodStart,
                state.periodStart.atOffset(ZoneOffset.UTC).plusMonths(1).toInstant());
    }

    /**
     * Get usage stats for all tenants (for admin dashboard).
     */
    public List<QuotaStats> allStats() {
        return tenants.keySet().stream().map(this::stats).toList();
    }

    /**
     * Reset a tenant's usage counter (admin action or period rollover).
     */
    public void reset(String tenantId) {
        TenantState state = getOrCreate(tenantId);
        state.usedTokens.set(0);
        state.usedInputTokens.set(0);
        state.usedOutputTokens.set(0);
        state.accumulatedCostMicros.set(0);
        state.periodStart = YearMonth.now().atDay(1).atStartOfDay().toInstant(ZoneOffset.UTC);
        log.infof("Quota reset for tenant '%s'", tenantId);
    }

    // ─────────────────────────────────────────────────────────────────────────

    private TenantState getOrCreate(String tenantId) {
        return tenants.computeIfAbsent(tenantId, k -> {
            TenantState s = new TenantState();
            s.periodStart = YearMonth.now().atDay(1).atStartOfDay()
                    .toInstant(ZoneOffset.UTC);
            return s;
        });
    }

    private long getBudget(String tenantId) {
        return mpConfig.getOptionalValue(
                "gollek.quota.tenant." + tenantId + ".monthly-tokens", Long.class)
                .orElse(defaultMonthlyTokens);
    }

    private double getCostPer1kInput(String tenantId) {
        return mpConfig.getOptionalValue(
                "gollek.quota.tenant." + tenantId + ".cost-per-1k-input", Double.class)
                .orElse(defaultCostPer1kInput);
    }

    private double getCostPer1kOutput(String tenantId) {
        return mpConfig.getOptionalValue(
                "gollek.quota.tenant." + tenantId + ".cost-per-1k-output", Double.class)
                .orElse(defaultCostPer1kOutput);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // State & value types
    // ─────────────────────────────────────────────────────────────────────────

    private static final class TenantState {
        volatile Instant periodStart;
        final AtomicLong usedTokens = new AtomicLong(0);
        final AtomicLong usedInputTokens = new AtomicLong(0);
        final AtomicLong usedOutputTokens = new AtomicLong(0);
        final AtomicLong accumulatedCostMicros = new AtomicLong(0);

        void resetIfNewPeriod() {
            Instant startOfMonth = YearMonth.now().atDay(1)
                    .atStartOfDay().toInstant(ZoneOffset.UTC);
            if (periodStart.isBefore(startOfMonth)) {
                // New billing period — reset counters atomically
                synchronized (this) {
                    if (periodStart.isBefore(startOfMonth)) {
                        usedTokens.set(0);
                        usedInputTokens.set(0);
                        usedOutputTokens.set(0);
                        accumulatedCostMicros.set(0);
                        periodStart = startOfMonth;
                    }
                }
            }
        }
    }

    /**
     * Quota statistics snapshot for a tenant.
     */
    public record QuotaStats(
            String tenantId,
            long usedTokens,
            long usedInputTokens,
            long usedOutputTokens,
            long budgetTokens,
            double accumulatedCost,
            Instant periodStart,
            Instant periodEnd) {

        /** Percentage of quota used (0.0–100.0). */
        public double usedPercent() {
            return budgetTokens > 0 ? usedTokens * 100.0 / budgetTokens : 0;
        }

        /** True when more than 90% of the quota is consumed. */
        public boolean isNearLimit() {
            return usedPercent() >= 90;
        }

        /** True when the budget is fully exhausted. */
        public boolean isExhausted() {
            return usedTokens >= budgetTokens;
        }
    }
}
