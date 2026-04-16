package tech.kayys.gollek.scheduler;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.kvcache.KVCacheConfig;
import tech.kayys.gollek.kvcache.KVCacheExhaustedException;
import tech.kayys.gollek.scheduler.ContinuousBatchingScheduler.ScheduledRequest;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ContinuousBatchingScheduler.
 */
class ContinuousBatchingSchedulerTest {

    private PagedKVCacheManager mockKVCacheManager;
    private ContinuousBatchingScheduler scheduler;

    @BeforeEach
    void setUp() {
        // Create mock KV cache manager with minimal memory footprint for tests
        mockKVCacheManager = new MockPagedKVCacheManager(128, 16, 1, 64);
        
        scheduler = ContinuousBatchingScheduler.builder()
            .maxBatchSize(8)
            .maxSequenceLength(64)
            .kvCacheManager(mockKVCacheManager)
            .scheduleIntervalMs(1) // Faster for tests
            .build();
    }

    @AfterEach
    void tearDown() {
        if (scheduler != null) {
            scheduler.close();
        }
        if (mockKVCacheManager != null) {
            mockKVCacheManager.close();
        }
    }

    @Test
    @DisplayName("Scheduler starts successfully")
    void testSchedulerStarts() {
        assertDoesNotThrow(() -> scheduler.start());
    }

    @Test
    @DisplayName("Submit request is accepted when scheduler is running")
    void testSubmitRequest() {
        scheduler.start();
        
        ScheduledRequest request = new ScheduledRequest(
            "req-1",
            "What is the capital of France?",
            100
        );
        
        assertTrue(scheduler.submit(request));
    }

    @Test
    @DisplayName("Submit request is rejected when scheduler is not running")
    void testSubmitWhenNotRunning() {
        ScheduledRequest request = new ScheduledRequest(
            "req-1",
            "What is the capital of France?",
            100
        );
        
        assertFalse(scheduler.submit(request));
    }

    @Test
    @DisplayName("Cancel pending request")
    void testCancelPendingRequest() {
        scheduler.start();
        
        // Fill running slots to force queuing
        for (int i = 0; i < 8; i++) {
            ScheduledRequest running = new ScheduledRequest(
                "running-" + i,
                "Prompt " + i,
                1000
            );
            scheduler.submit(running);
        }
        
        // This should go to pending queue
        ScheduledRequest pending = new ScheduledRequest(
            "pending-1",
            "Pending prompt",
            100
        );
        scheduler.submit(pending);
        
        // Cancel it
        assertTrue(scheduler.cancel("pending-1"));
    }

    @Test
    @DisplayName("Get metrics returns valid data")
    void testGetMetrics() {
        scheduler.start();
        
        ContinuousBatchingScheduler.SchedulerMetrics metrics = scheduler.getMetrics();
        
        assertNotNull(metrics);
        assertEquals(0, metrics.pendingRequests());
        assertEquals(0, metrics.runningRequests());
        assertEquals(0, metrics.totalProcessed());
        assertTrue(metrics.cacheUtilization() >= 0.0);
    }

    @Test
    @DisplayName("Scheduled request tracks lifecycle correctly")
    void testScheduledRequestLifecycle() {
        ScheduledRequest request = new ScheduledRequest(
            "req-1",
            "Test prompt here",
            50
        );
        
        assertFalse(request.isStarted());
        assertFalse(request.isCompleted());
        assertFalse(request.isCancelled());
        assertEquals(0, request.getGeneratedTokens());
        
        request.markStarted();
        assertTrue(request.isStarted());
        
        request.incrementGeneratedTokens();
        request.incrementGeneratedTokens();
        request.incrementGeneratedTokens();
        assertEquals(3, request.getGeneratedTokens());
        
        request.markCompleted();
        assertTrue(request.isCompleted());
    }

    @Test
    @DisplayName("Metrics calculation is correct")
    void testMetricsCalculation() {
        ContinuousBatchingScheduler.SchedulerMetrics metrics = 
            new ContinuousBatchingScheduler.SchedulerMetrics(
                5,      // pending
                10,     // running
                100,    // totalProcessed
                50000,  // totalTokensGenerated
                2,      // rejected
                0.75,   // avgBatchUtilization
                512,    // availableKvCacheBlocks
                1024    // totalKvCacheBlocks
            );
        
        assertEquals(5, metrics.pendingRequests());
        assertEquals(10, metrics.runningRequests());
        assertEquals(100, metrics.totalProcessed());
        assertEquals(50000, metrics.totalTokensGenerated());
        assertEquals(0.5, metrics.cacheUtilization(), 0.01);
    }

    @Test
    @DisplayName("Request with priority is created correctly")
    void testPriorityRequest() {
        ScheduledRequest highPriority = new ScheduledRequest(
            "req-high",
            "High priority prompt",
            100,
            1  // High priority
        );
        
        ScheduledRequest lowPriority = new ScheduledRequest(
            "req-low",
            "Low priority prompt",
            100,
            10  // Low priority
        );
        
        assertTrue(highPriority.getPriority().isPresent());
        assertEquals(1, highPriority.getPriority().get());
        assertEquals(10, lowPriority.getPriority().get());
    }

    @Test
    @DisplayName("Close scheduler multiple times is safe")
    void testMultipleClose() {
        scheduler.start();
        scheduler.close();
        scheduler.close();  // Should not throw
    }

    /**
     * Mock KV cache manager for testing without real off-heap memory.
     */
    static class MockPagedKVCacheManager extends PagedKVCacheManager {
        private int availableBlocks;
        private final int totalBlocks;

        MockPagedKVCacheManager(int totalBlocks, int blockSize, int numHeads, int headDim) {
            super(KVCacheConfig.builder()
                .totalBlocks(totalBlocks)
                .blockSize(blockSize)
                .numHeads(numHeads)
                .headDim(headDim)
                .build());
            this.totalBlocks = totalBlocks;
            this.availableBlocks = totalBlocks;
        }

        @Override
        public int getFreeBlockCount() {
            return availableBlocks;
        }

        public int totalBlocks() {
            return totalBlocks;
        }

        @Override
        public int blocksRequired(int numTokens) {
            return (numTokens + 15) / 16;  // blockSize=16
        }

        @Override
        public java.util.List<Integer> allocateForPrefill(String requestId, int numTokens) {
            int needed = blocksRequired(numTokens);
            if (needed > availableBlocks) {
                throw new KVCacheExhaustedException(
                    "Not enough blocks: need " + needed + ", have " + availableBlocks);
            }
            availableBlocks -= needed;
            return java.util.stream.IntStream.range(0, needed)
                .boxed()
                .collect(java.util.stream.Collectors.toList());
        }

        @Override
        public int freeRequest(String requestId) {
            // Simplified: just return some blocks
            availableBlocks = Math.min(availableBlocks + 16, totalBlocks);
            return 16;
        }
    }
}
