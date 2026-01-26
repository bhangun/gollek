package tech.kayys.golek.core.observability;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.metrics.Meter;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.context.Scope;

/**
 * Central observability manager for metrics, tracing, and logging
 */
public class ObservabilityManager {
    private final OpenTelemetry openTelemetry;
    private final Meter meter;
    private final Tracer tracer;
    private final MetricsRegistry metricsRegistry;
    private final LogAggregator logAggregator;

    public ObservabilityManager(OpenTelemetry openTelemetry) {
        this.openTelemetry = openTelemetry;
        this.meter = openTelemetry.getMeter("golek-core");
        this.tracer = openTelemetry.getTracer("golek-core");
        this.metricsRegistry = new MetricsRegistry(meter);
        this.logAggregator = new LogAggregator();
    }

    /**
     * Get the metrics registry
     */
    public MetricsRegistry getMetricsRegistry() {
        return metricsRegistry;
    }

    /**
     * Get the log aggregator
     */
    public LogAggregator getLogAggregator() {
        return logAggregator;
    }

    /**
     * Create a trace span for an operation
     */
    public <T> T traced(String operationName, SpanKind kind, Attributes attributes,
            SupplierWithException<T> operation) throws Exception {
        Span span = tracer.spanBuilder(operationName)
                .setSpanKind(kind)
                .setAllAttributes(attributes)
                .startSpan();

        try (Scope scope = span.makeCurrent()) {
            T result = operation.get();
            span.setStatus(io.opentelemetry.api.trace.StatusCode.OK);
            return result;
        } catch (Exception e) {
            span.recordException(e);
            span.setStatus(io.opentelemetry.api.trace.StatusCode.ERROR, e.getMessage());
            throw e;
        } finally {
            span.end();
        }
    }

    /**
     * Create a trace span for a runnable operation
     */
    public void traced(String operationName, SpanKind kind, Attributes attributes,
            RunnableWithException operation) throws Exception {
        Span span = tracer.spanBuilder(operationName)
                .setSpanKind(kind)
                .setAllAttributes(attributes)
                .startSpan();

        try (Scope scope = span.makeCurrent()) {
            operation.run();
            span.setStatus(io.opentelemetry.api.trace.StatusCode.OK);
        } catch (Exception e) {
            span.recordException(e);
            span.setStatus(io.opentelemetry.api.trace.StatusCode.ERROR, e.getMessage());
            throw e;
        } finally {
            span.end();
        }
    }

    /**
     * Record a metric
     */
    public void recordMetric(String metricName, double value, Attributes attributes) {
        metricsRegistry.record(metricName, value, attributes);
    }

    /**
     * Add a log entry
     */
    public void log(String level, String message, Attributes attributes) {
        logAggregator.addLog(level, message, attributes);
    }

    /**
     * Functional interface for operations that can throw exceptions
     */
    @FunctionalInterface
    public interface SupplierWithException<T> {
        T get() throws Exception;
    }

    /**
     * Functional interface for runnables that can throw exceptions
     */
    @FunctionalInterface
    public interface RunnableWithException {
        void run() throws Exception;
    }
}