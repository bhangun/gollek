package tech.kayys.gollek.engine.inference;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferenceObserver;

import org.jboss.logging.Logger;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Default observer implementation with metrics and tracing.
 */
@ApplicationScoped
public class DefaultInferenceObserver implements InferenceObserver {

    private static final Logger LOG = Logger.getLogger(DefaultInferenceObserver.class);

    @Inject
    MeterRegistry meterRegistry;

    @Inject
    Tracer tracer;

    // Track active spans per execution
    private final ConcurrentHashMap<String, Span> activeSpans = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Timer.Sample> activeTimers = new ConcurrentHashMap<>();

    @Override
    public void onStart(ExecutionContext context) {
        String executionId = context.token().getExecutionId();

        // Start trace span
        Span span = tracer.spanBuilder("inference.execute")
                .setAttribute("request.id", context.token().getRequestId())
                .setAttribute("tenant.id", context.requestContext().getRequestId())
                .setAttribute("execution.id", executionId)
                .startSpan();

        activeSpans.put(executionId, span);

        // Start timer
        Timer.Sample timer = Timer.start(meterRegistry);
        activeTimers.put(executionId, timer);

        LOG.debugf("Inference started: %s", executionId);
    }

    @Override
    public void onPhase(InferencePhase phase, ExecutionContext context) {
        String executionId = context.token().getExecutionId();

        Span parentSpan = activeSpans.get(executionId);
        if (parentSpan != null) {
            // Create phase span
            Span phaseSpan = tracer.spanBuilder("phase." + phase.name().toLowerCase())
                    .setParent(Context.current().with(parentSpan))
                    .setAttribute("phase", phase.name())
                    .setAttribute("phase.order", phase.getOrder())
                    .startSpan();

            // Store with phase-specific key
            activeSpans.put(executionId + ":" + phase.name(), phaseSpan);
        }

        // Record phase metric
        meterRegistry.counter(
                "inference.phase.started",
                "phase", phase.name(),
                "tenant", context.requestContext().getRequestId()).increment();
    }

    @Override
    public void onSuccess(ExecutionContext context) {
        String executionId = context.token().getExecutionId();

        // Stop timer
        Timer.Sample timer = activeTimers.remove(executionId);
        if (timer != null) {
            timer.stop(meterRegistry.timer(
                    "inference.duration",
                    "status", "success",
                    "tenant", context.requestContext().getRequestId()));
        }

        // Close all spans
        closeAllSpans(executionId);

        // Record success metric
        meterRegistry.counter(
                "inference.completed",
                "status", "success",
                "tenant", context.requestContext().getRequestId()).increment();

        LOG.infof("Inference completed: %s", executionId);
    }

    @Override
    public void onFailure(Throwable error, ExecutionContext context) {
        String executionId = context.token().getExecutionId();

        // Stop timer
        Timer.Sample timer = activeTimers.remove(executionId);
        if (timer != null) {
            timer.stop(meterRegistry.timer(
                    "inference.duration",
                    "status", "failure",
                    "tenant", context.requestContext().getRequestId()));
        }

        // Record error in span
        Span span = activeSpans.get(executionId);
        if (span != null) {
            span.recordException(error);
        }

        // Close all spans
        closeAllSpans(executionId);

        // Record failure metric
        meterRegistry.counter(
                "inference.completed",
                "status", "failure",
                "error.type", error.getClass().getSimpleName(),
                "tenant", context.requestContext().getRequestId()).increment();

        LOG.errorf(error, "Inference failed: %s", executionId);
    }

    @Override
    public void onInferenceChunk(Object chunk, ExecutionContext context) {
        // Optional: track streaming chunks
        meterRegistry.counter("inference.stream.chunk").increment();
    }

    @Override
    public void onStreamComplete(ExecutionContext context) {
        String executionId = context.token().getExecutionId();

        // Close span
        Span span = activeSpans.remove(executionId);
        if (span != null) {
            span.end();
        }

        // Record timer
        Timer.Sample timer = activeTimers.remove(executionId);
        if (timer != null) {
            timer.stop(meterRegistry.timer("inference.streaming.duration"));
        }

        LOG.debugf("Streaming completed: %s", executionId);
    }

    private void closeAllSpans(String executionId) {
        // Close all spans for this execution
        activeSpans.entrySet()
                .stream()
                .filter(entry -> entry.getKey().startsWith(executionId))
                .forEach(entry -> {
                    entry.getValue().end();
                    activeSpans.remove(entry.getKey());
                });
    }
}