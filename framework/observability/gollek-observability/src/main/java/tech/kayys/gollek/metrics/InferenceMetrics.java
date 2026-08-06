package tech.kayys.gollek.metrics;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.time.Duration;

/**
 * Metrics collection for inference operations.
 *
 * <h3>Key Metrics Exposed to Prometheus</h3>
 * <ul>
 *   <li>{@code gollek_inference_duration_seconds} – total request latency.</li>
 *   <li>{@code gollek_inference_ttft_seconds}     – Time-To-First-Token (TTFT).
 *       Used by KEDA to trigger pod scale-out when streaming latency degrades.</li>
 *   <li>{@code gollek_inference_tpot_seconds}     – Time-Per-Output-Token (TPOT).
 *       Average inter-token interval; used for throughput anomaly detection.</li>
 *   <li>{@code gollek_inference_requests_total}   – request counter by status.</li>
 *   <li>{@code gollek_inference_tokens_total}     – total tokens generated.</li>
 * </ul>
 */
@ApplicationScoped
public class InferenceMetrics {

        private final MeterRegistry registry;

        @Inject
        public InferenceMetrics(MeterRegistry registry) {
                this.registry = registry;
        }

        public void recordSuccess(
                        String model,
                        String requestId,
                        long durationNanos,
                        int tokensUsed) {
                Timer.builder("inference.duration")
                                .tag("model", model)
                                .tag("tenant", requestId)
                                .tag("status", "success")
                                .register(registry)
                                .record(Duration.ofNanos(durationNanos));

                Counter.builder("inference.requests")
                                .tag("model", model)
                                .tag("tenant", requestId)
                                .tag("status", "success")
                                .register(registry)
                                .increment();

                Counter.builder("inference.tokens")
                                .tag("model", model)
                                .tag("tenant", requestId)
                                .register(registry)
                                .increment(tokensUsed);
        }

        public void recordFailure(
                        String model,
                        String requestId,
                        String errorType) {
                Counter.builder("inference.requests")
                                .tag("model", model)
                                .tag("tenant", requestId)
                                .tag("status", "failed")
                                .tag("error", errorType)
                                .register(registry)
                                .increment();
        }

        public void recordInferenceChunk(String model, String tenantId) {
                Counter.builder("inference.stream.chunks")
                                .tag("model", model)
                                .tag("tenant", tenantId)
                                .register(registry)
                                .increment();
        }

        /**
         * Records the Time-To-First-Token (TTFT) for a streaming request.
         *
         * <p>TTFT is the duration from when the request was submitted to when
         * the first token was received from the provider.  It reflects queuing
         * delay + network RTT + model prefill time.  Export this metric to
         * Prometheus and configure a KEDA {@code ScaledObject} to trigger
         * horizontal pod autoscaling when the P99 TTFT exceeds a threshold.
         *
         * @param model      model name tag
         * @param tenantId   tenant / org identifier
         * @param ttftNanos  measured TTFT in nanoseconds
         */
        public void recordTtft(String model, String tenantId, long ttftNanos) {
                Timer.builder("inference.ttft")
                                .description("Time-To-First-Token (TTFT) per streaming request")
                                .tag("model", model)
                                .tag("tenant", tenantId)
                                .publishPercentiles(0.5, 0.9, 0.99)
                                .register(registry)
                                .record(Duration.ofNanos(ttftNanos));
        }

        /**
         * Records the Time-Per-Output-Token (TPOT) for a streaming request.
         *
         * <p>TPOT = (total_generation_time) / (output_tokens).  It reflects
         * decoding throughput. Anomaly detection rules (e.g., Prometheus
         * alerting rules or KEDA external scalers) can watch this metric to
         * identify model degradation or GPU saturation.
         *
         * @param model           model name tag
         * @param tenantId        tenant / org identifier
         * @param tpotNanos       average inter-token latency in nanoseconds
         * @param outputTokens    total number of tokens generated
         */
        public void recordTpot(String model, String tenantId, long tpotNanos, int outputTokens) {
                Timer.builder("inference.tpot")
                                .description("Time-Per-Output-Token (TPOT) averaged over the response")
                                .tag("model", model)
                                .tag("tenant", tenantId)
                                .publishPercentiles(0.5, 0.9, 0.99)
                                .register(registry)
                                .record(Duration.ofNanos(tpotNanos));

                // Also track raw token throughput as a counter for KEDA
                io.micrometer.core.instrument.Counter.builder("inference.output.tokens")
                                .description("Total output tokens generated – use for KEDA throughput scaling")
                                .tag("model", model)
                                .tag("tenant", tenantId)
                                .register(registry)
                                .increment(outputTokens);
        }

        /**
         * Convenience method: derives TPOT from total generation time and token count.
         *
         * @param model              model tag
         * @param tenantId           tenant tag
         * @param generationNanos    total time spent generating tokens (nanos)
         * @param outputTokens       number of tokens produced
         */
        public void recordTpotFromGeneration(String model, String tenantId,
                                             long generationNanos, int outputTokens) {
                if (outputTokens <= 0) return;
                long tpotNanos = generationNanos / outputTokens;
                recordTpot(model, tenantId, tpotNanos, outputTokens);
        }
}