package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.inference.libtorch.ContinuousBatchingManager.BatchRequest;
import tech.kayys.gollek.inference.libtorch.ContinuousBatchingManager.Priority;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for priority-based batch ordering in
 * {@link ContinuousBatchingManager}.
 */
class ContinuousBatchingPriorityTest {

    // ── Priority enum tests ──────────────────────────────────────────

    @Test
    void priorityFromStringValues() {
        assertEquals(Priority.CRITICAL, Priority.fromString("critical"));
        assertEquals(Priority.HIGH, Priority.fromString("high"));
        assertEquals(Priority.NORMAL, Priority.fromString("normal"));
        assertEquals(Priority.LOW, Priority.fromString("low"));
    }

    @Test
    void priorityFromStringCaseInsensitive() {
        assertEquals(Priority.CRITICAL, Priority.fromString("CRITICAL"));
        assertEquals(Priority.HIGH, Priority.fromString("High"));
    }

    @Test
    void priorityFromStringNullReturnsNormal() {
        assertEquals(Priority.NORMAL, Priority.fromString(null));
    }

    @Test
    void priorityFromStringUnknownReturnsNormal() {
        assertEquals(Priority.NORMAL, Priority.fromString("unknown"));
        assertEquals(Priority.NORMAL, Priority.fromString(""));
    }

    @Test
    void priorityLevelOrdering() {
        assertTrue(Priority.CRITICAL.level() < Priority.HIGH.level());
        assertTrue(Priority.HIGH.level() < Priority.NORMAL.level());
        assertTrue(Priority.NORMAL.level() < Priority.LOW.level());
    }

    // ── BatchRequest ordering tests ──────────────────────────────────

    @Test
    void batchRequestsSortByPriority() {
        BatchRequest low = makeBatchRequest("r1", Priority.LOW);
        BatchRequest high = makeBatchRequest("r2", Priority.HIGH);
        BatchRequest critical = makeBatchRequest("r3", Priority.CRITICAL);
        BatchRequest normal = makeBatchRequest("r4", Priority.NORMAL);

        List<BatchRequest> list = new ArrayList<>(List.of(low, high, critical, normal));
        Collections.sort(list);

        assertEquals(Priority.CRITICAL, list.get(0).priority());
        assertEquals(Priority.HIGH, list.get(1).priority());
        assertEquals(Priority.NORMAL, list.get(2).priority());
        assertEquals(Priority.LOW, list.get(3).priority());
    }

    @Test
    void samePriorityFifoOrdering() throws InterruptedException {
        BatchRequest first = makeBatchRequest("r1", Priority.NORMAL);
        Thread.sleep(1); // ensure distinct enqueuedAt
        BatchRequest second = makeBatchRequest("r2", Priority.NORMAL);

        List<BatchRequest> list = new ArrayList<>(List.of(second, first));
        Collections.sort(list);

        // First enqueued should come first
        assertEquals("r1", list.get(0).requestId());
        assertEquals("r2", list.get(1).requestId());
    }

    @Test
    void backwardCompatibleConstructorDefaultsToNormal() {
        BatchRequest req = new BatchRequest(
                "tenant", "model", Path.of("/tmp/model.pt"), "req-1",
                null, t -> null, new CompletableFuture<>());
        assertEquals(Priority.NORMAL, req.priority());
    }

    @Test
    void batchManagerRejectsWhenNotRunning() {
        ContinuousBatchingManager manager = new ContinuousBatchingManager();
        BatchRequest req = makeBatchRequest("r1", Priority.NORMAL);
        CompletableFuture<?> future = manager.enqueue(req);
        assertTrue(future.isCompletedExceptionally(),
                "Should fail when batching manager is not running");
    }

    @Test
    void initialStateCountsAreZero() {
        ContinuousBatchingManager manager = new ContinuousBatchingManager();
        assertEquals(0, manager.getBatchCount());
        assertEquals(0, manager.getTotalBatchedRequests());
        assertEquals(0, manager.getQueueDepth());
        assertFalse(manager.isRunning());
    }

    // ── Helpers ───────────────────────────────────────────────────────

    private BatchRequest makeBatchRequest(String requestId, Priority priority) {
        return new BatchRequest(
                "tenant1", "model1", Path.of("/tmp/model.pt"), requestId,
                priority, System.nanoTime(),
                null, null, t -> null, new CompletableFuture<>());
    }
}
