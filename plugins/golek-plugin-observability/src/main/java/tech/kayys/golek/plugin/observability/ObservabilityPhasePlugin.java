/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.observability;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.core.plugin.InferencePhasePlugin;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.PluginException;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Plugin for recording inference observability metrics.
 * <p>
 * Bound to {@link InferencePhase#OBSERVABILITY}.
 * Records token usage, latency per phase, tool usage, trace context, and error tracking.
 * Designed for integration with OpenTelemetry, Prometheus, and logging frameworks.
 */
@ApplicationScoped
public class ObservabilityPhasePlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(ObservabilityPhasePlugin.class);
    private static final String PLUGIN_ID = "tech.kayys/observability";

    private boolean enabled = true;
    private boolean logMetrics = true;
    private Map<String, Object> config = new HashMap<>();

    // In-memory metrics counters (for export to external systems)
    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalInputTokens = new AtomicLong(0);
    private final AtomicLong totalOutputTokens = new AtomicLong(0);
    private final AtomicLong totalErrors = new AtomicLong(0);
    private final Map<String, AtomicLong> toolUsageCounters = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> phaseLatencies = new ConcurrentHashMap<>();

    private final List<MetricsExporter> exporters = new ArrayList<>();

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.OBSERVABILITY;
    }

    @Override
    public int order() {
        return 10;
    }

    @Override
    public void initialize(PluginContext context) {
        context.getConfig("enabled").ifPresent(v -> this.enabled = Boolean.parseBoolean(v));
        context.getConfig("logMetrics").ifPresent(v -> this.logMetrics = Boolean.parseBoolean(v));
        LOG.infof("Initialized %s (logMetrics: %s)", PLUGIN_ID, logMetrics);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        return enabled;
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        totalRequests.incrementAndGet();

        // Build trace for this request
        var trace = new InferenceTrace(
                context.getVariable("requestId", String.class).orElse("unknown"),
                Instant.now());

        // Record token usage from response
        InferenceResponse response = context.getVariable("response", InferenceResponse.class)
                .orElse(null);

        if (response != null) {
            int inputTokens = response.getInputTokens();
            int outputTokens = response.getOutputTokens();
            totalInputTokens.addAndGet(inputTokens);
            totalOutputTokens.addAndGet(outputTokens);
            trace.setInputTokens(inputTokens);
            trace.setOutputTokens(outputTokens);

            if (logMetrics) {
                LOG.infof("Inference metrics — input: %d tokens, output: %d tokens, model: %s",
                        inputTokens, outputTokens,
                        context.getVariable("modelId", String.class).orElse("unknown"));
            }
        }

        // Record tool usage
        @SuppressWarnings("unchecked")
        List<String> usedTools = (List<String>) context.getVariable("usedTools", List.class)
                .orElse(Collections.emptyList());
        for (String tool : usedTools) {
            toolUsageCounters.computeIfAbsent(tool, k -> new AtomicLong(0)).incrementAndGet();
            trace.addToolUsage(tool);
        }

        // Record reasoning steps
        int steps = context.getVariable("totalReasoningSteps", Integer.class).orElse(1);
        trace.setReasoningSteps(steps);

        // Record errors
        Throwable error = context.getVariable("error", Throwable.class).orElse(null);
        if (error != null) {
            totalErrors.incrementAndGet();
            trace.setError(error.getMessage());
        }

        // Record phase latencies from context
        @SuppressWarnings("unchecked")
        Map<String, Long> latencies = (Map<String, Long>) context.getVariable("phaseLatencies", Map.class)
                .orElse(Collections.emptyMap());

        for (Map.Entry<String, Long> entry : latencies.entrySet()) {
            phaseLatencies.computeIfAbsent(entry.getKey(), k -> new AtomicLong(0))
                    .addAndGet(entry.getValue());
            trace.addPhaseLatency(entry.getKey(), Duration.ofMillis(entry.getValue()));
        }

        // Store trace
        context.putVariable("inferenceTrace", trace);

        // Export to registered exporters
        for (MetricsExporter exporter : exporters) {
            try {
                exporter.export(trace);
            } catch (Exception e) {
                LOG.warnf(e, "Failed to export metrics to %s", exporter.getClass().getSimpleName());
            }
        }
    }

    /**
     * Register a metrics exporter.
     */
    public void addExporter(MetricsExporter exporter) {
        exporters.add(exporter);
    }

    /**
     * Get current aggregate metrics snapshot.
     */
    public Map<String, Object> getMetricsSnapshot() {
        var snapshot = new HashMap<String, Object>();
        snapshot.put("totalRequests", totalRequests.get());
        snapshot.put("totalInputTokens", totalInputTokens.get());
        snapshot.put("totalOutputTokens", totalOutputTokens.get());
        snapshot.put("totalErrors", totalErrors.get());
        snapshot.put("toolUsage", Map.copyOf(toolUsageCounters));
        snapshot.put("phaseLatencies", Map.copyOf(phaseLatencies));
        return snapshot;
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        this.logMetrics = (Boolean) newConfig.getOrDefault("logMetrics", true);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of(
                "enabled", enabled,
                "logMetrics", logMetrics);
    }
}
