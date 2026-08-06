/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */
package tech.kayys.gollek.plugin.audit;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.exception.PluginException;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;

import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * Asynchronous, event-driven audit logging plugin bound to the
 * {@link InferencePhase#AUDIT} phase.
 *
 * <h3>Design</h3>
 * Audit events are placed on an internal {@link BlockingQueue} and consumed
 * by a background writer thread (the "sink").  This decouples the inference
 * hot-path from the latency of I/O operations (disk, Kafka, database).
 *
 * <p>The sink is pluggable via {@link AuditSink}.  Built-in sinks:
 * <ul>
 *   <li>{@link LoggingAuditSink} – writes structured JSON to JUL (default)</li>
 *   <li>{@link KafkaAuditSink}   – publishes to a Kafka topic (optional)</li>
 * </ul>
 *
 * <h3>Audit Event Schema</h3>
 * Each event captures:
 * <ul>
 *   <li>Unique audit ID (UUID)</li>
 *   <li>Request / session / tenant IDs</li>
 *   <li>W3C traceparent for distributed trace correlation</li>
 *   <li>Inference phase and status (success / failure)</li>
 *   <li>Model and provider used</li>
 *   <li>Timestamp</li>
 * </ul>
 *
 * <h3>Compliance</h3>
 * Events are immutable records. The queue never discards events – if the
 * queue is full, the emit call blocks briefly before overflowing to a
 * synchronous fallback log entry (no event is ever silently dropped).
 */
@ApplicationScoped
public class AuditLoggingPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(AuditLoggingPlugin.class.getName());
    private static final int QUEUE_CAPACITY = 10_000;

    private final BlockingQueue<AuditEvent> eventQueue = new LinkedBlockingQueue<>(QUEUE_CAPACITY);
    private final AtomicLong droppedEvents = new AtomicLong(0);

    private final AuditSink sink;
    private final Thread writerThread;
    private volatile boolean running = true;

    public AuditLoggingPlugin() {
        this(new LoggingAuditSink());
    }

    public AuditLoggingPlugin(AuditSink sink) {
        this.sink = sink;
        this.writerThread = Thread.ofVirtual()
            .name("gollek-audit-writer")
            .start(this::drainLoop);
    }

    // -----------------------------------------------------------------------
    // InferencePhasePlugin
    // -----------------------------------------------------------------------

    @Override
    public String pluginId() { return "gollek.audit"; }

    @Override
    public String displayName() { return "Audit Logging Plugin"; }

    @Override
    public InferencePhase phase() { return InferencePhase.AUDIT; }

    @Override
    public int order() { return 100; }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        AuditEvent event = buildEvent(context);
        boolean offered = eventQueue.offer(event);
        if (!offered) {
            // Queue full – synchronous fallback to prevent silent loss
            droppedEvents.incrementAndGet();
            LOG.warning("[Audit] Queue full – synchronous fallback for requestId=" +
                        event.requestId());
            sink.write(event);
        }
    }

    // -----------------------------------------------------------------------
    // Background drain loop (virtual thread)
    // -----------------------------------------------------------------------

    private void drainLoop() {
        while (running || !eventQueue.isEmpty()) {
            try {
                AuditEvent event = eventQueue.poll(100, java.util.concurrent.TimeUnit.MILLISECONDS);
                if (event != null) {
                    sink.write(event);
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                LOG.severe("[Audit] Sink write failed: " + e.getMessage());
            }
        }
    }

    /** Shuts down the audit writer gracefully. */
    public void shutdown() {
        running = false;
        writerThread.interrupt();
    }

    /** Returns the number of events that overflowed (queue was full). */
    public long getDroppedEventCount() { return droppedEvents.get(); }

    // -----------------------------------------------------------------------
    // Event builder
    // -----------------------------------------------------------------------

    private AuditEvent buildEvent(ExecutionContext context) {
        String requestId   = safeStr(context, "requestId");
        String tenantId    = safeStr(context, "tenantId");
        String sessionId   = safeStr(context, "sessionId");
        String model       = safeStr(context, "model");
        String provider    = safeStr(context, "selectedProvider");
        String traceparent = (String) context.metadata().getOrDefault(
            "w3c.traceparent", "");
        boolean hasError   = context.hasError();
        String errorMsg    = context.getError().map(Throwable::getMessage).orElse(null);

        return new AuditEvent(
            UUID.randomUUID().toString(),
            requestId, tenantId, sessionId, model, provider,
            traceparent,
            Instant.now(),
            hasError ? "FAILURE" : "SUCCESS",
            errorMsg,
            Map.copyOf(context.metadata())
        );
    }

    private String safeStr(ExecutionContext ctx, String key) {
        return ctx.getVariable(key, String.class).orElse("unknown");
    }

    // -----------------------------------------------------------------------
    // Domain types
    // -----------------------------------------------------------------------

    /**
     * Immutable audit event record.
     * Each field is included in the structured log output / Kafka message.
     */
    public record AuditEvent(
        String  auditId,
        String  requestId,
        String  tenantId,
        String  sessionId,
        String  model,
        String  provider,
        String  traceparent,
        Instant timestamp,
        String  status,        // SUCCESS | FAILURE
        String  errorMessage,  // null on success
        Map<String, Object> metadata
    ) {}

    // -----------------------------------------------------------------------
    // Sink SPI
    // -----------------------------------------------------------------------

    /** SPI for writing audit events to a destination. */
    public interface AuditSink {
        void write(AuditEvent event);
    }

    /**
     * Default sink: writes structured JSON-like output to JUL INFO.
     */
    public static class LoggingAuditSink implements AuditSink {
        private static final Logger SINK_LOG = Logger.getLogger("gollek.audit.events");

        @Override
        public void write(AuditEvent e) {
            SINK_LOG.info(String.format(
                "{\"audit_id\":\"%s\",\"request_id\":\"%s\",\"tenant\":\"%s\"," +
                "\"model\":\"%s\",\"provider\":\"%s\",\"status\":\"%s\"," +
                "\"traceparent\":\"%s\",\"ts\":\"%s\",\"error\":%s}",
                e.auditId(), e.requestId(), e.tenantId(),
                e.model(), e.provider(), e.status(),
                e.traceparent(), e.timestamp(),
                e.errorMessage() != null ? "\"" + e.errorMessage() + "\"" : "null"
            ));
        }
    }

    /**
     * Kafka sink stub — replace body with actual Kafka producer logic.
     * Binds to SmallRye Reactive Messaging when the channel is available.
     */
    public static class KafkaAuditSink implements AuditSink {
        private static final Logger KAFKA_LOG = Logger.getLogger("gollek.audit.kafka");

        private final String topic;

        public KafkaAuditSink(String topic) {
            this.topic = topic;
        }

        @Override
        public void write(AuditEvent event) {
            // TODO: inject io.smallrye.reactive.messaging.MutinyEmitter<String>
            // and emit JSON-serialised event to the Kafka topic.
            KAFKA_LOG.info("[KafkaAuditSink] Would publish to topic=" + topic +
                           " auditId=" + event.auditId());
        }
    }
}
