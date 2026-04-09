package tech.kayys.gollek.engine.inference;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import tech.kayys.gollek.spi.batch.BatchConfig;
import tech.kayys.gollek.spi.batch.BatchInferenceRequest;
import tech.kayys.gollek.spi.batch.BatchResponse;
import tech.kayys.gollek.spi.batch.BatchStrategy;
import tech.kayys.gollek.spi.inference.InferenceEngine;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.Message;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link DefaultBatchScheduler} covering all three batching
 * strategies.
 *
 * <p>
 * No Quarkus container is required — same pattern as
 * {@link InferenceOrchestratorTest}.
 */
@ExtendWith(MockitoExtension.class)
class DefaultBatchSchedulerTest {

    @Mock
    private InferenceEngine orchestrator;

    private static InferenceResponse responseFor(String requestId) {
        return InferenceResponse.builder()
                .requestId(requestId)
                .content("ok")
                .build();
    }

    private static InferenceRequest requestFor(String id) {
        return InferenceRequest.builder()
                .requestId(id)
                .model("test-model")
                .message(new Message(Message.Role.USER, "hi"))
                .build();
    }

    // ── STATIC strategy tests ─────────────────────────────────────────────────

    @Test
    @Timeout(5)
    void staticBatching_dispatchesOnlyAfterBatchIsFull() throws Exception {
        // Arrange: maxBatchSize=3, orchestrator answers immediately
        BatchConfig cfg = new BatchConfig(BatchStrategy.STATIC, 3, Duration.ZERO, 1, 3, 128, false);
        when(orchestrator.executeAsync(anyString(), any()))
                .thenAnswer(inv -> {
                    InferenceRequest req = inv.getArgument(1);
                    return Uni.createFrom().item(responseFor(req.getRequestId()));
                });

        DefaultBatchScheduler scheduler = new DefaultBatchScheduler(orchestrator, cfg);
        try {
            // Submit 3 requests
            List<CompletableFuture<InferenceResponse>> futures = List.of(
                    scheduler.submit(requestFor("r1")),
                    scheduler.submit(requestFor("r2")),
                    scheduler.submit(requestFor("r3")));

            // All three should complete
            for (CompletableFuture<InferenceResponse> f : futures) {
                InferenceResponse response = f.get(4, TimeUnit.SECONDS);
                assertThat(response).isNotNull();
                assertThat(response.getContent()).isEqualTo("ok");
            }
        } finally {
            scheduler.stop();
        }
    }

    @Test
    @Timeout(5)
    void staticBatching_flush_forcesDispatchBeforeBatchIsFull() throws Exception {
        // Arrange: maxBatchSize=10, so batch would never fill organically
        BatchConfig cfg = new BatchConfig(BatchStrategy.STATIC, 10, Duration.ZERO, 1, 4, 128, false);
        when(orchestrator.executeAsync(anyString(), any()))
                .thenAnswer(inv -> {
                    InferenceRequest req = inv.getArgument(1);
                    return Uni.createFrom().item(responseFor(req.getRequestId()));
                });

        DefaultBatchScheduler scheduler = new DefaultBatchScheduler(orchestrator, cfg);
        try {
            CompletableFuture<InferenceResponse> future = scheduler.submit(requestFor("flush-req"));

            // Give the dispatch loop a moment to pick up the request then flush
            Thread.sleep(50);
            scheduler.flush();

            InferenceResponse response = future.get(4, TimeUnit.SECONDS);
            assertThat(response).isNotNull();
            assertThat(response.getContent()).isEqualTo("ok");
        } finally {
            scheduler.stop();
        }
    }

    // ── DYNAMIC strategy tests ────────────────────────────────────────────────

    @Test
    @Timeout(5)
    void dynamicBatching_dispatchesBeforeFullOnTimeout() throws Exception {
        // Arrange: maxBatchSize=10 but maxWaitTime=50ms → single request dispatched
        // after timeout
        BatchConfig cfg = new BatchConfig(BatchStrategy.DYNAMIC, 10, Duration.ofMillis(50), 2, 4, 128, false);
        when(orchestrator.executeAsync(anyString(), any()))
                .thenAnswer(inv -> {
                    InferenceRequest req = inv.getArgument(1);
                    return Uni.createFrom().item(responseFor(req.getRequestId()));
                });

        DefaultBatchScheduler scheduler = new DefaultBatchScheduler(orchestrator, cfg);
        try {
            CompletableFuture<InferenceResponse> future = scheduler.submit(requestFor("lonely"));

            // Should complete after ~50ms timeout even though batch is not full
            InferenceResponse response = future.get(4, TimeUnit.SECONDS);
            assertThat(response).isNotNull();
        } finally {
            scheduler.stop();
        }
    }

    @Test
    @Timeout(5)
    void dynamicBatching_dispatchesImmediatelyWhenBatchFull() throws Exception {
        // Arrange: maxBatchSize=3, maxWaitTime=5s — batch should dispatch well before
        // timeout
        BatchConfig cfg = new BatchConfig(BatchStrategy.DYNAMIC, 3, Duration.ofSeconds(5), 2, 3, 128, false);
        AtomicInteger callCount = new AtomicInteger(0);
        when(orchestrator.executeAsync(anyString(), any()))
                .thenAnswer(inv -> {
                    callCount.incrementAndGet();
                    InferenceRequest req = inv.getArgument(1);
                    return Uni.createFrom().item(responseFor(req.getRequestId()));
                });

        DefaultBatchScheduler scheduler = new DefaultBatchScheduler(orchestrator, cfg);
        try {
            List<CompletableFuture<InferenceResponse>> futures = List.of(
                    scheduler.submit(requestFor("d1")),
                    scheduler.submit(requestFor("d2")),
                    scheduler.submit(requestFor("d3")));

            for (CompletableFuture<InferenceResponse> f : futures) {
                assertThat(f.get(4, TimeUnit.SECONDS)).isNotNull();
            }
            // All 3 requests dispatched
            assertThat(callCount.get()).isEqualTo(3);
        } finally {
            scheduler.stop();
        }
    }

    // ── CONTINUOUS strategy tests ─────────────────────────────────────────────

    @Test
    @Timeout(5)
    void continuousBatching_completedSlotsAcceptNextRequests() throws Exception {
        // Arrange: maxBatchSize=2, submit 4 requests — should all complete via
        // continuous iteration
        BatchConfig cfg = new BatchConfig(BatchStrategy.CONTINUOUS, 2, Duration.ofMillis(10), 2, 2, 128, false);
        when(orchestrator.executeAsync(anyString(), any()))
                .thenAnswer(inv -> {
                    InferenceRequest req = inv.getArgument(1);
                    return Uni.createFrom().item(responseFor(req.getRequestId()));
                });

        DefaultBatchScheduler scheduler = new DefaultBatchScheduler(orchestrator, cfg);
        try {
            List<CompletableFuture<InferenceResponse>> futures = List.of(
                    scheduler.submit(requestFor("c1")),
                    scheduler.submit(requestFor("c2")),
                    scheduler.submit(requestFor("c3")),
                    scheduler.submit(requestFor("c4")));

            for (CompletableFuture<InferenceResponse> f : futures) {
                assertThat(f.get(4, TimeUnit.SECONDS)).isNotNull();
            }
        } finally {
            scheduler.stop();
        }
    }

    // ── submitBatch / BatchResponse tests ─────────────────────────────────────

    @Test
    @Timeout(5)
    void submitBatch_returnsBatchResponseWithAllResults() throws Exception {
        BatchConfig cfg = BatchConfig.defaultDynamic();
        when(orchestrator.executeAsync(anyString(), any()))
                .thenAnswer(inv -> {
                    InferenceRequest req = inv.getArgument(1);
                    return Uni.createFrom().item(responseFor(req.getRequestId()));
                });

        DefaultBatchScheduler scheduler = new DefaultBatchScheduler(orchestrator, cfg);
        try {
            BatchInferenceRequest batch = BatchInferenceRequest.builder()
                    .modelId("test-model")
                    .inputs(List.of(
                            Map.of("prompt", "hello"),
                            Map.of("prompt", "world")))
                    .build();

            BatchResponse response = scheduler.submitBatch(batch)
                    .get(4, TimeUnit.SECONDS);

            assertThat(response).isNotNull();
            assertThat(response.results()).hasSize(2);
            assertThat(response.successCount()).isEqualTo(2);
            assertThat(response.failureCount()).isZero();
            assertThat(response.metrics()).isNotNull();
        } finally {
            scheduler.stop();
        }
    }

    // ── Hot config reload ─────────────────────────────────────────────────────

    @Test
    @Timeout(5)
    void setConfig_hotReloadsStrategy() throws Exception {
        BatchConfig initial = BatchConfig.defaultDynamic();
        when(orchestrator.executeAsync(anyString(), any()))
                .thenAnswer(inv -> {
                    InferenceRequest req = inv.getArgument(1);
                    return Uni.createFrom().item(responseFor(req.getRequestId()));
                });

        DefaultBatchScheduler scheduler = new DefaultBatchScheduler(orchestrator, initial);
        try {
            assertThat(scheduler.getConfig().strategy()).isEqualTo(BatchStrategy.DYNAMIC);

            scheduler.setConfig(BatchConfig.defaultContinuous());
            assertThat(scheduler.getConfig().strategy()).isEqualTo(BatchStrategy.CONTINUOUS);

            // Verify scheduler still works after reload
            CompletableFuture<InferenceResponse> future = scheduler.submit(requestFor("hot-reload"));
            assertThat(future.get(4, TimeUnit.SECONDS)).isNotNull();
        } finally {
            scheduler.stop();
        }
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    @Test
    @Timeout(5)
    void snapshot_returnsNonNullMetrics() throws Exception {
        BatchConfig cfg = BatchConfig.defaultDynamic();
        when(orchestrator.executeAsync(anyString(), any()))
                .thenAnswer(inv -> {
                    InferenceRequest req = inv.getArgument(1);
                    return Uni.createFrom().item(responseFor(req.getRequestId()));
                });

        DefaultBatchScheduler scheduler = new DefaultBatchScheduler(orchestrator, cfg);
        try {
            scheduler.submit(requestFor("m1")).get(4, TimeUnit.SECONDS);
            var metrics = scheduler.snapshot();
            assertThat(metrics.totalRequests()).isGreaterThan(0);
            assertThat(metrics.currentQueueDepth()).isGreaterThanOrEqualTo(0);
        } finally {
            scheduler.stop();
        }
    }
}
