package tech.kayys.gollek.observability;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.time.Duration;

/**
 * Metrics collection for inference operations
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

        public void recordStreamChunk(String model, String requestId) {
                Counter.builder("inference.stream.chunks")
                                .tag("model", model)
                                .tag("tenant", requestId)
                                .register(registry)
                                .increment();
        }
}