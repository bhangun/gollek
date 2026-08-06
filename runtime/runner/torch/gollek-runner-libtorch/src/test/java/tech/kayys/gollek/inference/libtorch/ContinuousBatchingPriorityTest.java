package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.runtime.inference.batch.BatchRequest;
import tech.kayys.gollek.runtime.inference.batch.ContinuousBatchScheduler;
import tech.kayys.gollek.spi.inference.Priority;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for priority-based batch ordering in {@link ContinuousBatchScheduler}.
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

        assertEquals(Priority.CRITICAL, list.get(0).priority);
        assertEquals(Priority.HIGH, list.get(1).priority);
        assertEquals(Priority.NORMAL, list.get(2).priority);
        assertEquals(Priority.LOW, list.get(3).priority);
    }

    @Test
    void samePriorityFifoOrdering() throws InterruptedException {
        // Enqueue times are nanos from System.nanoTime() in constructor
        BatchRequest first = makeBatchRequest("r1", Priority.NORMAL);
        Thread.sleep(1);
        BatchRequest second = makeBatchRequest("r2", Priority.NORMAL);

        List<BatchRequest> list = new ArrayList<>(List.of(second, first));
        Collections.sort(list);

        // First enqueued should come first
        assertEquals("tenant-r1", list.get(0).tenantId);
        assertEquals("tenant-r2", list.get(1).tenantId);
    }

    @Test
    void defaultPriorityIsNormal() {
        BatchRequest req = new BatchRequest(List.of(1, 2, 3), null, null, 10);
        assertEquals(Priority.NORMAL, req.priority);
    }

    @Test
    void initialStateCountsAreZero() {
        ContinuousBatchScheduler manager = new ContinuousBatchScheduler(10);
        assertEquals(0, manager.getBatchCount());
        assertEquals(0, manager.getTotalBatchedRequests());
        assertEquals(0, manager.getQueueDepth());
        assertFalse(manager.isRunning());
    }

    // ── Helpers ───────────────────────────────────────────────────────

    private BatchRequest makeBatchRequest(String requestId, Priority priority) {
        // We use tenantId to store the requestId for verification in tests
        return new BatchRequest(
                "tenant-" + requestId, 
                List.of(1, 2, 3), 
                null, 
                null, 
                100, 
                priority);
    }
}
