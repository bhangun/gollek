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

package tech.kayys.golek.observability;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.metrics.Meter;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Scope;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Central observability manager for metrics, tracing, and logging.
 * 
 * Provides unified API for:
 * - Distributed tracing with OpenTelemetry
 * - Metrics collection and recording
 * - Log aggregation
 * 
 * Thread-safe and resource-efficient.
 */
public class ObservabilityManager implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(ObservabilityManager.class.getName());
    private static final String COMPONENT_NAME = "golek-core";
    private static final Attributes EMPTY_ATTRIBUTES = Attributes.empty();

    private final OpenTelemetry openTelemetry;
    private final Meter meter;
    private final Tracer tracer;
    private final MetricsRegistry metricsRegistry;
    private final LogAggregator logAggregator;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Creates a new ObservabilityManager with default configuration.
     *
     * @param openTelemetry the OpenTelemetry instance to use for tracing and
     *                      metrics
     * @throws NullPointerException if openTelemetry is null
     */
    public ObservabilityManager(OpenTelemetry openTelemetry) {
        this(openTelemetry, 1000);
    }

    /**
     * Creates a new ObservabilityManager with custom log buffer size.
     *
     * @param openTelemetry the OpenTelemetry instance to use for tracing and
     *                      metrics
     * @param maxLogEntries maximum number of log entries to buffer
     * @throws NullPointerException     if openTelemetry is null
     * @throws IllegalArgumentException if maxLogEntries is less than 1
     */
    public ObservabilityManager(OpenTelemetry openTelemetry, int maxLogEntries) {
        this.openTelemetry = Objects.requireNonNull(openTelemetry, "openTelemetry cannot be null");
        if (maxLogEntries < 1) {
            throw new IllegalArgumentException("maxLogEntries must be at least 1");
        }
        this.meter = openTelemetry.getMeter(COMPONENT_NAME);
        this.tracer = openTelemetry.getTracer(COMPONENT_NAME);
        this.metricsRegistry = new MetricsRegistry(meter);
        this.logAggregator = new LogAggregator(maxLogEntries);
    }

    /**
     * Get the metrics registry for direct metric operations.
     *
     * @return the metrics registry
     * @throws IllegalStateException if manager is closed
     */
    public MetricsRegistry getMetricsRegistry() {
        checkNotClosed();
        return metricsRegistry;
    }

    /**
     * Get the log aggregator for direct log operations.
     *
     * @return the log aggregator
     * @throws IllegalStateException if manager is closed
     */
    public LogAggregator getLogAggregator() {
        checkNotClosed();
        return logAggregator;
    }

    /**
     * Create a trace span for an operation that returns a result.
     * Automatically manages span lifecycle and status.
     *
     * @param operationName name of the operation
     * @param operation     the operation to execute
     * @return the result of the operation
     * @throws Exception             if the operation throws
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if operationName or operation is null
     */
    public <T> T traced(String operationName, SupplierWithException<T> operation) throws Exception {
        return traced(operationName, SpanKind.INTERNAL, EMPTY_ATTRIBUTES, operation);
    }

    /**
     * Create a trace span for a runnable operation.
     * Automatically manages span lifecycle and status.
     *
     * @param operationName name of the operation
     * @param operation     the operation to execute
     * @throws Exception             if the operation throws
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if operationName or operation is null
     */
    public void traced(String operationName, RunnableWithException operation) throws Exception {
        traced(operationName, SpanKind.INTERNAL, EMPTY_ATTRIBUTES, operation);
    }

    /**
     * Create a trace span for an operation with custom span kind and attributes.
     *
     * @param operationName name of the operation
     * @param kind          the span kind
     * @param attributes    span attributes
     * @param operation     the operation to execute
     * @return the result of the operation
     * @throws Exception             if the operation throws
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if any required parameter is null
     */
    public <T> T traced(String operationName, SpanKind kind, Attributes attributes,
            SupplierWithException<T> operation) throws Exception {
        checkNotClosed();
        Objects.requireNonNull(operationName, "operationName cannot be null");
        Objects.requireNonNull(kind, "kind cannot be null");
        Objects.requireNonNull(attributes, "attributes cannot be null");
        Objects.requireNonNull(operation, "operation cannot be null");

        Span span = tracer.spanBuilder(operationName)
                .setSpanKind(kind)
                .setAllAttributes(attributes)
                .startSpan();

        try (Scope scope = span.makeCurrent()) {
            T result = operation.get();
            span.setStatus(StatusCode.OK);
            return result;
        } catch (Exception e) {
            recordSpanException(span, e);
            throw e;
        } finally {
            span.end();
        }
    }

    /**
     * Create a trace span for a runnable operation with custom span kind and
     * attributes.
     *
     * @param operationName name of the operation
     * @param kind          the span kind
     * @param attributes    span attributes
     * @param operation     the operation to execute
     * @throws Exception             if the operation throws
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if any required parameter is null
     */
    public void traced(String operationName, SpanKind kind, Attributes attributes,
            RunnableWithException operation) throws Exception {
        checkNotClosed();
        Objects.requireNonNull(operationName, "operationName cannot be null");
        Objects.requireNonNull(kind, "kind cannot be null");
        Objects.requireNonNull(attributes, "attributes cannot be null");
        Objects.requireNonNull(operation, "operation cannot be null");

        Span span = tracer.spanBuilder(operationName)
                .setSpanKind(kind)
                .setAllAttributes(attributes)
                .startSpan();

        try (Scope scope = span.makeCurrent()) {
            operation.run();
            span.setStatus(StatusCode.OK);
        } catch (Exception e) {
            recordSpanException(span, e);
            throw e;
        } finally {
            span.end();
        }
    }

    /**
     * Record a metric value.
     *
     * @param metricName name of the metric
     * @param value      the value to record
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if metricName is null
     */
    public void recordMetric(String metricName, double value) {
        recordMetric(metricName, value, EMPTY_ATTRIBUTES);
    }

    /**
     * Record a metric value with attributes.
     *
     * @param metricName name of the metric
     * @param value      the value to record
     * @param attributes metric attributes
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if metricName or attributes is null
     */
    public void recordMetric(String metricName, double value, Attributes attributes) {
        checkNotClosed();
        Objects.requireNonNull(metricName, "metricName cannot be null");
        Objects.requireNonNull(attributes, "attributes cannot be null");
        try {
            metricsRegistry.record(metricName, value, attributes);
        } catch (Exception e) {
            logger.log(Level.WARNING, "Failed to record metric: " + metricName, e);
        }
    }

    /**
     * Increment a counter metric.
     *
     * @param metricName name of the counter
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if metricName is null
     */
    public void incrementMetric(String metricName) {
        incrementMetric(metricName, EMPTY_ATTRIBUTES);
    }

    /**
     * Increment a counter metric with attributes.
     *
     * @param metricName name of the counter
     * @param attributes metric attributes
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if metricName or attributes is null
     */
    public void incrementMetric(String metricName, Attributes attributes) {
        checkNotClosed();
        Objects.requireNonNull(metricName, "metricName cannot be null");
        Objects.requireNonNull(attributes, "attributes cannot be null");
        try {
            metricsRegistry.increment(metricName, attributes);
        } catch (Exception e) {
            logger.log(Level.WARNING, "Failed to increment metric: " + metricName, e);
        }
    }

    /**
     * Add a value to a counter metric.
     *
     * @param metricName name of the counter
     * @param value      the value to add
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if metricName is null
     */
    public void addMetric(String metricName, long value) {
        addMetric(metricName, value, EMPTY_ATTRIBUTES);
    }

    /**
     * Add a value to a counter metric with attributes.
     *
     * @param metricName name of the counter
     * @param value      the value to add
     * @param attributes metric attributes
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if metricName or attributes is null
     */
    public void addMetric(String metricName, long value, Attributes attributes) {
        checkNotClosed();
        Objects.requireNonNull(metricName, "metricName cannot be null");
        Objects.requireNonNull(attributes, "attributes cannot be null");
        try {
            metricsRegistry.add(metricName, value, attributes);
        } catch (Exception e) {
            logger.log(Level.WARNING, "Failed to add metric: " + metricName, e);
        }
    }

    /**
     * Add a log entry.
     *
     * @param level   log level
     * @param message log message
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if level or message is null
     */
    public void log(String level, String message) {
        log(level, message, EMPTY_ATTRIBUTES);
    }

    /**
     * Add a log entry with attributes.
     *
     * @param level      log level
     * @param message    log message
     * @param attributes log attributes
     * @throws IllegalStateException if manager is closed
     * @throws NullPointerException  if level, message, or attributes is null
     */
    public void log(String level, String message, Attributes attributes) {
        checkNotClosed();
        Objects.requireNonNull(level, "level cannot be null");
        Objects.requireNonNull(message, "message cannot be null");
        Objects.requireNonNull(attributes, "attributes cannot be null");
        try {
            logAggregator.addLog(level, message, attributes);
        } catch (Exception e) {
            logger.log(Level.WARNING, "Failed to add log entry", e);
        }
    }

    /**
     * Check if the manager is closed.
     *
     * @return true if closed, false otherwise
     */
    public boolean isClosed() {
        return closed.get();
    }

    /**
     * Close the manager and cleanup resources.
     * Flushes and closes OpenTelemetry, clears logs.
     * Safe to call multiple times.
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            try {
                logAggregator.clearLogs();

                // Force flush and close OpenTelemetry to ensure metrics and traces are exported
                if (openTelemetry instanceof io.opentelemetry.sdk.OpenTelemetrySdk) {
                    ((io.opentelemetry.sdk.OpenTelemetrySdk) openTelemetry).close();
                }

                logger.fine("ObservabilityManager closed successfully");
            } catch (Exception e) {
                logger.log(Level.WARNING, "Error closing ObservabilityManager", e);
            }
        }
    }

    /**
     * Check if manager is closed and throw if it is.
     *
     * @throws IllegalStateException if closed
     */
    private void checkNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("ObservabilityManager is closed");
        }
    }

    /**
     * Record an exception in a span with proper error handling.
     *
     * @param span the span to record the exception in
     * @param e    the exception to record
     */
    private void recordSpanException(Span span, Exception e) {
        try {
            span.recordException(e);
            String errorMessage = e.getMessage() != null ? e.getMessage() : e.getClass().getName();
            span.setStatus(StatusCode.ERROR, errorMessage);
        } catch (Exception ex) {
            logger.log(Level.WARNING, "Failed to record exception in span", ex);
        }
    }

    /**
     * Functional interface for operations that return a result and can throw
     * exceptions.
     */
    @FunctionalInterface
    public interface SupplierWithException<T> {
        T get() throws Exception;
    }

    /**
     * Functional interface for runnable operations that can throw exceptions.
     */
    @FunctionalInterface
    public interface RunnableWithException {
        void run() throws Exception;
    }
}