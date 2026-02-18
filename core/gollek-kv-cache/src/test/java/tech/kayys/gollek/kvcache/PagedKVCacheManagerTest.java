package tech.kayys.gollek.kvcache;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for {@link PagedKVCacheManager}.
 */
class PagedKVCacheManagerTest {

    private KVCacheConfig config;
    private PagedKVCacheManager manager;

    @BeforeEach
    void setUp() {
        config = KVCacheConfig.builder()
                .blockSize(4)       // Small block size for testing
                .totalBlocks(8)     // Small pool for testing
                .numLayers(2)       // Small model for testing
                .numHeads(2)
                .headDim(4)
                .useGpu(false)
                .build();
        manager = new PagedKVCacheManager(config);
    }

    @AfterEach
    void tearDown() {
        if (manager != null) {
            manager.close();
        }
    }

    // --- Allocation Tests ---

    @Test
    @DisplayName("Allocate blocks for a prefill request")
    void allocateForPrefill_basic() {
        List<Integer> blocks = manager.allocateForPrefill("req-1", 10);

        // 10 tokens / 4 blockSize = 3 blocks (ceil)
        assertThat(blocks).hasSize(3);
        assertThat(manager.getFreeBlockCount()).isEqualTo(5); // 8 - 3
        assertThat(manager.getAllocatedBlockCount()).isEqualTo(3);
        assertThat(manager.getActiveRequestCount()).isEqualTo(1);
        assertThat(manager.getTokenCount("req-1")).isEqualTo(10);
    }

    @Test
    @DisplayName("Allocate exactly one block when tokens fit")
    void allocateForPrefill_exactFit() {
        List<Integer> blocks = manager.allocateForPrefill("req-1", 4);

        assertThat(blocks).hasSize(1);
        assertThat(manager.getFreeBlockCount()).isEqualTo(7);
    }

    @Test
    @DisplayName("Allocate blocks for multiple requests")
    void allocateForPrefill_multipleRequests() {
        manager.allocateForPrefill("req-1", 4);  // 1 block
        manager.allocateForPrefill("req-2", 8);  // 2 blocks
        manager.allocateForPrefill("req-3", 3);  // 1 block

        assertThat(manager.getActiveRequestCount()).isEqualTo(3);
        assertThat(manager.getAllocatedBlockCount()).isEqualTo(4);
        assertThat(manager.getFreeBlockCount()).isEqualTo(4);
    }

    @Test
    @DisplayName("Throw KVCacheExhaustedException when pool is exhausted")
    void allocateForPrefill_exhausted() {
        // Allocate all 8 blocks (8 blocks × 4 tokens = 32 tokens)
        manager.allocateForPrefill("req-1", 32);

        assertThatThrownBy(() -> manager.allocateForPrefill("req-2", 1))
                .isInstanceOf(KVCacheExhaustedException.class)
                .hasMessageContaining("exhausted");
    }

    @Test
    @DisplayName("Reject duplicate request ID")
    void allocateForPrefill_duplicateRequest() {
        manager.allocateForPrefill("req-1", 4);

        assertThatThrownBy(() -> manager.allocateForPrefill("req-1", 4))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("already has allocated blocks");
    }

    @Test
    @DisplayName("Reject zero or negative token count")
    void allocateForPrefill_invalidTokens() {
        assertThatThrownBy(() -> manager.allocateForPrefill("req-1", 0))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> manager.allocateForPrefill("req-1", -5))
                .isInstanceOf(IllegalArgumentException.class);
    }

    // --- Decode Extension Tests ---

    @Test
    @DisplayName("Extend allocation during decode within existing block")
    void extendForDecode_withinBlock() {
        manager.allocateForPrefill("req-1", 2); // 1 block, 2/4 slots used

        // Tokens 3 and 4 fit in the existing block
        int blockId = manager.extendForDecode("req-1");
        assertThat(blockId).isGreaterThanOrEqualTo(0);
        assertThat(manager.getAllocatedBlockCount()).isEqualTo(1); // No new block needed
    }

    @Test
    @DisplayName("Extend allocation triggers new block when current is full")
    void extendForDecode_newBlock() {
        manager.allocateForPrefill("req-1", 4); // 1 block, 4/4 slots used

        // Token 5 needs a new block
        int newBlockId = manager.extendForDecode("req-1");
        assertThat(newBlockId).isGreaterThanOrEqualTo(0);
        assertThat(manager.getAllocatedBlockCount()).isEqualTo(2);
        assertThat(manager.getBlockTable("req-1")).hasSize(2);
    }

    @Test
    @DisplayName("appendToken returns true only when new block is needed")
    void appendToken_tracking() {
        manager.allocateForPrefill("req-1", 3); // 1 block, 3/4 used

        assertThat(manager.appendToken("req-1")).isFalse(); // Token 4, still fits
        assertThat(manager.appendToken("req-1")).isTrue();  // Token 5, new block
        assertThat(manager.getAllocatedBlockCount()).isEqualTo(2);
    }

    // --- Deallocation Tests ---

    @Test
    @DisplayName("Free request returns blocks to pool")
    void freeRequest_basic() {
        manager.allocateForPrefill("req-1", 12); // 3 blocks
        assertThat(manager.getFreeBlockCount()).isEqualTo(5);

        int freed = manager.freeRequest("req-1");
        assertThat(freed).isEqualTo(3);
        assertThat(manager.getFreeBlockCount()).isEqualTo(8);
        assertThat(manager.getActiveRequestCount()).isEqualTo(0);
    }

    @Test
    @DisplayName("Free request allows re-allocation")
    void freeRequest_thenReallocate() {
        manager.allocateForPrefill("req-1", 32); // All 8 blocks
        assertThat(manager.getFreeBlockCount()).isEqualTo(0);

        manager.freeRequest("req-1");
        assertThat(manager.getFreeBlockCount()).isEqualTo(8);

        // Can allocate again
        manager.allocateForPrefill("req-2", 16);
        assertThat(manager.getFreeBlockCount()).isEqualTo(4);
    }

    @Test
    @DisplayName("Free unknown request returns 0")
    void freeRequest_unknown() {
        int freed = manager.freeRequest("nonexistent");
        assertThat(freed).isEqualTo(0);
    }

    // --- Block Table Tests ---

    @Test
    @DisplayName("Block table preserves order")
    void getBlockTable_order() {
        List<Integer> blocks = manager.allocateForPrefill("req-1", 12); // 3 blocks
        List<Integer> table = manager.getBlockTable("req-1");

        assertThat(table).isEqualTo(blocks);
    }

    @Test
    @DisplayName("Block table grows with decode extension")
    void getBlockTable_growsWithDecode() {
        manager.allocateForPrefill("req-1", 4); // 1 block
        assertThat(manager.getBlockTable("req-1")).hasSize(1);

        manager.extendForDecode("req-1"); // new block
        assertThat(manager.getBlockTable("req-1")).hasSize(2);
    }

    @Test
    @DisplayName("Throw on getting block table for unknown request")
    void getBlockTable_unknown() {
        assertThatThrownBy(() -> manager.getBlockTable("nonexistent"))
                .isInstanceOf(IllegalArgumentException.class);
    }

    // --- Metrics Tests ---

    @Test
    @DisplayName("Utilization reflects allocation state")
    void utilization() {
        assertThat(manager.getUtilization()).isEqualTo(0.0);

        manager.allocateForPrefill("req-1", 16); // 4 blocks
        assertThat(manager.getUtilization()).isEqualTo(0.5);

        manager.allocateForPrefill("req-2", 16); // 4 more blocks
        assertThat(manager.getUtilization()).isEqualTo(1.0);
    }

    @Test
    @DisplayName("Stats snapshot is accurate")
    void stats() {
        manager.allocateForPrefill("req-1", 8); // 2 blocks
        manager.freeRequest("req-1");
        manager.allocateForPrefill("req-2", 4); // 1 block

        var stats = manager.getStats();
        assertThat(stats.totalBlocks()).isEqualTo(8);
        assertThat(stats.freeBlocks()).isEqualTo(7);
        assertThat(stats.allocatedBlocks()).isEqualTo(1);
        assertThat(stats.activeRequests()).isEqualTo(1);
        assertThat(stats.totalAllocations()).isEqualTo(3); // 2 + 1
        assertThat(stats.totalDeallocations()).isEqualTo(2);
    }

    @Test
    @DisplayName("hasCapacity checks correctly")
    void hasCapacity() {
        assertThat(manager.hasCapacity(8)).isTrue();
        assertThat(manager.hasCapacity(9)).isFalse();

        manager.allocateForPrefill("req-1", 16); // 4 blocks
        assertThat(manager.hasCapacity(4)).isTrue();
        assertThat(manager.hasCapacity(5)).isFalse();
    }

    // --- Max Blocks Per Request ---

    @Test
    @DisplayName("Enforce max blocks per request limit")
    void maxBlocksPerRequest() {
        manager.close();

        config = KVCacheConfig.builder()
                .blockSize(4)
                .totalBlocks(8)
                .numLayers(2)
                .numHeads(2)
                .headDim(4)
                .maxBlocksPerRequest(2) // Max 2 blocks = 8 tokens
                .build();
        manager = new PagedKVCacheManager(config);

        // 9 tokens needs 3 blocks > max 2
        assertThatThrownBy(() -> manager.allocateForPrefill("req-1", 9))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("max blocks per request");
    }

    // --- Concurrency Tests ---

    @Test
    @DisplayName("Concurrent allocations and deallocations are safe")
    void concurrentAllocAndFree() throws Exception {
        manager.close();

        config = KVCacheConfig.builder()
                .blockSize(4)
                .totalBlocks(100) // Larger pool for concurrency test
                .numLayers(2)
                .numHeads(2)
                .headDim(4)
                .build();
        manager = new PagedKVCacheManager(config);

        int numThreads = 10;
        int opsPerThread = 20;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch latch = new CountDownLatch(numThreads);
        AtomicInteger errors = new AtomicInteger(0);

        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    for (int op = 0; op < opsPerThread; op++) {
                        String reqId = "thread-" + threadId + "-op-" + op;
                        try {
                            manager.allocateForPrefill(reqId, 2); // 1 block each
                            Thread.sleep(1); // Simulate work
                            manager.freeRequest(reqId);
                        } catch (KVCacheExhaustedException e) {
                            // Expected under high contention
                        }
                    }
                } catch (Exception e) {
                    errors.incrementAndGet();
                } finally {
                    latch.countDown();
                }
            });
        }

        latch.await(10, TimeUnit.SECONDS);
        executor.shutdown();

        assertThat(errors.get()).isZero();
        // After all threads complete, all blocks should be free
        assertThat(manager.getFreeBlockCount()).isEqualTo(100);
        assertThat(manager.getActiveRequestCount()).isEqualTo(0);
    }
}
