/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */
package tech.kayys.gollek.plugin.observability;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferenceObserver;
import tech.kayys.gollek.spi.inference.InferencePhase;

import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Default implementation of {@link InferenceObserver} that:
 * <ul>
 *   <li>Records per-phase latency timers via Micrometer.</li>
 *   <li>Stores W3C Trace Context (traceparent / tracestate) in the
 *       {@link ExecutionContext} metadata so that downstream components
 *       (e.g., HTTP clients calling LLM providers) can inject the headers
 *       into outbound requests.</li>
 *   <li>Links child spans to the parent trace by propagating span IDs through
 *       the {@code traceparent} metadata key.</li>
 * </ul>
 *
 * <h3>W3C Trace Context Format</h3>
 * {@code traceparent: 00-<traceId>-<parentSpanId>-<flags>}
 * <p>
 * Flags: {@code 01} = sampled, {@code 00} = not sampled.
 *
 * @see <a href="https://www.w3.org/TR/trace-context/">W3C Trace Context</a>
 */
@ApplicationScoped
public class DefaultInferenceObserver implements InferenceObserver {

    private static final Logger LOG = Logger.getLogger(DefaultInferenceObserver.class.getName());

    /** Micrometer metric name prefix. */
    private static final String METRIC_PREFIX = "gollek.inference";

    /** Context metadata key: W3C traceparent header. */
    public static final String META_TRACEPARENT = "w3c.traceparent";

    /** Context metadata key: W3C tracestate header. */
    public static final String META_TRACESTATE = "w3c.tracestate";

    /** Context metadata key: current span ID. */
    public static final String META_SPAN_ID = "w3c.spanId";

    /** Context variable key: phase start epoch millis. */
    private static final String VAR_PHASE_START = "_phaseStartMs";

    /** Context variable key: request start epoch millis. */
    private static final String VAR_REQUEST_START = "_requestStartMs";

    /** W3C trace context version byte. */
    private static final String W3C_VERSION = "00";

    /** Sampling flag: always sampled (01). */
    private static final String FLAG_SAMPLED = "01";

    private final MeterRegistry registry;

    /** In-flight span IDs: requestId -> active spanId */
    private final Map<String, String> activeSpans = new ConcurrentHashMap<>();

    @Inject
    public DefaultInferenceObserver(MeterRegistry registry) {
        this.registry = registry;
    }

    // -----------------------------------------------------------------------
    // InferenceObserver lifecycle
    // -----------------------------------------------------------------------

    @Override
    public void onStart(ExecutionContext context) {
        long nowMs = Instant.now().toEpochMilli();
        context.putVariable(VAR_REQUEST_START, nowMs);

        String traceId = resolveTraceId(context);
        String spanId  = generateSpanId();
        String requestId = safeRequestId(context);

        activeSpans.put(requestId, spanId);

        // Build traceparent: 00-<traceId>-<spanId>-01
        String traceparent = buildTraceparent(traceId, spanId);

        // Write into context metadata so pipeline stages & HTTP clients can read it
        context.putMetadata(META_TRACEPARENT, traceparent);
        context.putMetadata(META_SPAN_ID, spanId);

        LOG.fine(() -> String.format(
            "[Tracing] Request started | requestId=%s traceparent=%s", requestId, traceparent));
    }

    @Override
    public void onPhase(InferencePhase phase, ExecutionContext context) {
        long nowMs = Instant.now().toEpochMilli();
        context.putVariable(VAR_PHASE_START, nowMs);

        // Generate a child span for this phase, with current span as parent
        String parentSpanId = (String) context.metadata().get(META_SPAN_ID);
        String childSpanId  = generateSpanId();
        String traceId      = resolveTraceId(context);

        String childTraceparent = buildTraceparent(traceId, childSpanId);
        // Keep the phase-level traceparent in metadata (updated per phase)
        context.putMetadata(META_TRACEPARENT, childTraceparent);
        context.putMetadata(META_SPAN_ID, childSpanId);

        LOG.fine(() -> String.format(
            "[Tracing] Phase=%s | childSpanId=%s parentSpanId=%s",
            phase.getDisplayName(), childSpanId, parentSpanId));
    }

    @Override
    public void onSuccess(ExecutionContext context) {
        long durationMs = elapsedSinceRequest(context);
        String requestId = safeRequestId(context);
        String model     = safeModel(context);

        Timer.builder(METRIC_PREFIX + ".duration")
             .tag("model", model)
             .tag("status", "success")
             .register(registry)
             .record(durationMs, TimeUnit.MILLISECONDS);

        LOG.fine(() -> String.format(
            "[Tracing] Request succeeded | requestId=%s model=%s durationMs=%d",
            requestId, model, durationMs));

        cleanup(context, requestId);
    }

    @Override
    public void onFailure(Throwable error, ExecutionContext context) {
        long durationMs = elapsedSinceRequest(context);
        String requestId = safeRequestId(context);
        String model     = safeModel(context);
        String errorType = error.getClass().getSimpleName();

        Timer.builder(METRIC_PREFIX + ".duration")
             .tag("model", model)
             .tag("status", "error")
             .tag("error_type", errorType)
             .register(registry)
             .record(durationMs, TimeUnit.MILLISECONDS);

        LOG.warning(String.format(
            "[Tracing] Request failed | requestId=%s model=%s durationMs=%d error=%s",
            requestId, model, durationMs, errorType));

        cleanup(context, requestId);
    }

    @Override
    public void onInferenceChunk(Object chunk, ExecutionContext context) {
        // Emit a streaming-chunk counter tagged with the active trace context
        String traceparent = (String) context.metadata().getOrDefault(META_TRACEPARENT, "");
        io.micrometer.core.instrument.Counter.builder(METRIC_PREFIX + ".stream.chunks")
             .tag("model", safeModel(context))
             .tag("traceparent_prefix", traceparent.length() > 20 ? traceparent.substring(0, 20) : traceparent)
             .register(registry)
             .increment();
    }

    @Override
    public void onStreamComplete(ExecutionContext context) {
        long durationMs = elapsedSinceRequest(context);
        String model    = safeModel(context);

        Timer.builder(METRIC_PREFIX + ".stream.duration")
             .tag("model", model)
             .tag("status", "complete")
             .register(registry)
             .record(durationMs, TimeUnit.MILLISECONDS);

        cleanup(context, safeRequestId(context));
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /**
     * Resolves the trace ID from the execution context.
     * Uses existing {@code requestId} as trace ID when no explicit trace is present.
     */
    private String resolveTraceId(ExecutionContext context) {
        // Check if the incoming request already carries a traceparent
        String incoming = (String) context.metadata().get(META_TRACEPARENT);
        if (incoming != null && incoming.length() >= 55) {
            // Extract trace ID from traceparent: 00-<traceId(32)>-<spanId(16)>-<flags(2)>
            return incoming.substring(3, 35);
        }
        // Fall back to requestId padded to 32 hex chars
        String requestId = safeRequestId(context).replaceAll("[^a-fA-F0-9]", "0");
        while (requestId.length() < 32) requestId = "0" + requestId;
        return requestId.substring(0, 32).toLowerCase();
    }

    /**
     * Generates a new 16-character (8-byte) span ID using secure random.
     */
    private String generateSpanId() {
        long val = java.util.concurrent.ThreadLocalRandom.current().nextLong();
        return String.format("%016x", val);
    }

    /**
     * Builds a W3C traceparent string.
     */
    private String buildTraceparent(String traceId, String spanId) {
        return W3C_VERSION + "-" + traceId + "-" + spanId + "-" + FLAG_SAMPLED;
    }

    private long elapsedSinceRequest(ExecutionContext context) {
        Object startMs = context.variables().get(VAR_REQUEST_START);
        if (startMs instanceof Long l) {
            return Instant.now().toEpochMilli() - l;
        }
        return 0L;
    }

    private String safeRequestId(ExecutionContext context) {
        try {
            return context.token().requestId() != null
                ? context.token().requestId()
                : "unknown";
        } catch (Exception e) {
            return "unknown";
        }
    }

    private String safeModel(ExecutionContext context) {
        try {
            return context.getVariable("model", String.class).orElse("unknown");
        } catch (Exception e) {
            return "unknown";
        }
    }

    private void cleanup(ExecutionContext context, String requestId) {
        activeSpans.remove(requestId);
        context.variables().remove(VAR_REQUEST_START);
        context.variables().remove(VAR_PHASE_START);
    }
}
