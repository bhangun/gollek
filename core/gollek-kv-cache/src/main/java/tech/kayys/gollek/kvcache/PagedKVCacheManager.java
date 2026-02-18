package tech.kayys.gollek.kvcache;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Paged KV-Cache Manager — the heart of PagedAttention.
 * <p>
 * Manages a pool of fixed-size physical memory blocks and maps them to
 * inference requests. Instead of allocating contiguous memory for the
 * maximum possible sequence length, blocks are allocated on-demand as
 * tokens are generated.
 * <p>
 * <b>Key benefits:</b>
 * <ul>
 *   <li>Zero fragmentation — max waste per request is {@code blockSize - 1} tokens</li>
 *   <li>Shared prefix — multiple requests can share blocks for common system prompts</li>
 *   <li>Dynamic growth — blocks are allocated as decode progresses</li>
 *   <li>Instant reclamation — blocks return to pool immediately on request completion</li>
 * </ul>
 * <p>
 * Thread-safe: uses concurrent data structures for the free pool and request mappings.
 * The block table (mapping of request ID → physical block IDs) is passed to the
 * native PagedAttention kernel via {@link PhysicalBlockPool#packBlockTable(int[])}.
 *
 * @see PhysicalBlockPool
 * @see KVCacheConfig
 */
public class PagedKVCacheManager implements AutoCloseable {

    private final PhysicalBlockPool blockPool;
    private final KVCacheConfig config;

    /** Pool of free block IDs */
    private final Deque<Integer> freeBlocks = new ConcurrentLinkedDeque<>();

    /** Mapping: requestId → ordered list of allocated block IDs */
    private final Map<String, List<Integer>> requestBlocks = new ConcurrentHashMap<>();

    /** Mapping: requestId → number of tokens currently stored */
    private final Map<String, Integer> requestTokenCounts = new ConcurrentHashMap<>();

    // -- Metrics --
    private final AtomicLong totalAllocations = new AtomicLong();
    private final AtomicLong totalDeallocations = new AtomicLong();

    /**
     * Create a new PagedKVCacheManager with the given configuration.
     * Pre-allocates the physical block pool and initializes the free list.
     *
     * @param config the KV-Cache configuration
     */
    public PagedKVCacheManager(KVCacheConfig config) {
        this.config = config;
        this.blockPool = new PhysicalBlockPool(config);

        // Initialize free pool with all block IDs
        for (int i = 0; i < config.getTotalBlocks(); i++) {
            freeBlocks.add(i);
        }
    }

    // ---- Allocation ----

    /**
     * Allocate blocks for a new request's initial prompt (Prefill phase).
     * Allocates exactly enough blocks to hold the given number of prompt tokens.
     *
     * @param requestId the unique request identifier
     * @param numTokens the number of prompt tokens to allocate for
     * @return the allocated block IDs
     * @throws KVCacheExhaustedException if not enough free blocks
     * @throws IllegalArgumentException  if numTokens ≤ 0
     * @throws IllegalStateException     if requestId is already allocated
     */
    public List<Integer> allocateForPrefill(String requestId, int numTokens) {
        if (numTokens <= 0) {
            throw new IllegalArgumentException("numTokens must be > 0, got: " + numTokens);
        }
        if (requestBlocks.containsKey(requestId)) {
            throw new IllegalStateException("Request already has allocated blocks: " + requestId);
        }

        int blocksNeeded = blocksRequired(numTokens);
        enforceMaxBlocksPerRequest(blocksNeeded);

        List<Integer> allocated = allocateBlocks(blocksNeeded);
        requestBlocks.put(requestId, new ArrayList<>(allocated));
        requestTokenCounts.put(requestId, numTokens);
        totalAllocations.addAndGet(blocksNeeded);

        return Collections.unmodifiableList(allocated);
    }

    /**
     * Extend a request's block allocation for the Decode phase.
     * Called when the current last block is full and a new token is being generated.
     *
     * @param requestId the request identifier
     * @return the newly allocated block ID
     * @throws KVCacheExhaustedException if no free blocks available
     * @throws IllegalArgumentException  if requestId is not found
     */
    public int extendForDecode(String requestId) {
        List<Integer> blocks = requestBlocks.get(requestId);
        if (blocks == null) {
            throw new IllegalArgumentException("Unknown request: " + requestId);
        }

        int currentTokens = requestTokenCounts.getOrDefault(requestId, 0);
        int currentCapacity = blocks.size() * config.getBlockSize();

        // Check if the current block still has room
        if (currentTokens < currentCapacity) {
            // No new block needed, just increment token count
            requestTokenCounts.put(requestId, currentTokens + 1);
            return blocks.getLast();
        }

        // Need a new block
        enforceMaxBlocksPerRequest(blocks.size() + 1);

        Integer newBlock = freeBlocks.pollFirst();
        if (newBlock == null) {
            throw new KVCacheExhaustedException(
                    "KV-Cache pool exhausted. Free: 0, Requested: 1. " +
                    "Consider increasing totalBlocks or evicting idle requests.");
        }

        blocks.add(newBlock);
        requestTokenCounts.put(requestId, currentTokens + 1);
        totalAllocations.incrementAndGet();

        return newBlock;
    }

    /**
     * Record that a new token has been added to the request's sequence.
     * Returns true if a new block was needed (and allocated), false otherwise.
     *
     * @param requestId the request identifier
     * @return true if a new block was allocated
     */
    public boolean appendToken(String requestId) {
        List<Integer> blocks = requestBlocks.get(requestId);
        if (blocks == null) {
            throw new IllegalArgumentException("Unknown request: " + requestId);
        }

        int currentTokens = requestTokenCounts.getOrDefault(requestId, 0);
        int currentCapacity = blocks.size() * config.getBlockSize();

        if (currentTokens < currentCapacity) {
            // Still room in the current block
            requestTokenCounts.put(requestId, currentTokens + 1);
            return false;
        }

        // Need to extend
        extendForDecode(requestId);
        return true;
    }

    // ---- Deallocation ----

    /**
     * Free all blocks allocated to a request.
     * Called when inference is complete (EOS token) or on error/cancellation.
     *
     * @param requestId the request identifier
     * @return the number of blocks freed
     */
    public int freeRequest(String requestId) {
        List<Integer> blocks = requestBlocks.remove(requestId);
        requestTokenCounts.remove(requestId);

        if (blocks == null || blocks.isEmpty()) {
            return 0;
        }

        // Return all blocks to the free pool
        for (int blockId : blocks) {
            freeBlocks.addLast(blockId);
        }

        totalDeallocations.addAndGet(blocks.size());
        return blocks.size();
    }

    // ---- Block Table Access ----

    /**
     * Get the block table (ordered list of physical block IDs) for a request.
     * This is the mapping the PagedAttention kernel needs to locate KV data.
     *
     * @param requestId the request identifier
     * @return unmodifiable list of block IDs in order
     */
    public List<Integer> getBlockTable(String requestId) {
        List<Integer> blocks = requestBlocks.get(requestId);
        if (blocks == null) {
            throw new IllegalArgumentException("Unknown request: " + requestId);
        }
        return Collections.unmodifiableList(blocks);
    }

    /**
     * Get the block table as a native int array MemorySegment for passing
     * to the PagedAttention CUDA kernel.
     *
     * @param requestId the request identifier
     * @return MemorySegment containing block IDs as native ints
     */
    public MemorySegment getBlockTableNative(String requestId) {
        List<Integer> blocks = getBlockTable(requestId);
        return blockPool.packBlockTable(
                blocks.stream().mapToInt(Integer::intValue).toArray()
        );
    }

    // ---- Query / Metrics ----

    /**
     * Get the number of free blocks remaining.
     */
    public int getFreeBlockCount() {
        return freeBlocks.size();
    }

    /**
     * Get the number of allocated (in-use) blocks.
     */
    public int getAllocatedBlockCount() {
        return config.getTotalBlocks() - freeBlocks.size();
    }

    /**
     * Get the total number of active requests.
     */
    public int getActiveRequestCount() {
        return requestBlocks.size();
    }

    /**
     * Get the number of tokens currently stored for a request.
     *
     * @param requestId the request identifier
     * @return token count, or -1 if request is not found
     */
    public int getTokenCount(String requestId) {
        return requestTokenCounts.getOrDefault(requestId, -1);
    }

    /**
     * Get pool utilization as a percentage (0.0 to 1.0).
     */
    public double getUtilization() {
        return (double) getAllocatedBlockCount() / config.getTotalBlocks();
    }

    /**
     * Check if the pool has capacity for at least the given number of blocks.
     */
    public boolean hasCapacity(int blocksNeeded) {
        return freeBlocks.size() >= blocksNeeded;
    }

    /**
     * Get the underlying physical block pool.
     * Used by the PagedAttention kernel binding to access raw memory.
     */
    public PhysicalBlockPool getBlockPool() {
        return blockPool;
    }

    /**
     * Get the configuration.
     */
    public KVCacheConfig getConfig() {
        return config;
    }

    /**
     * Get a snapshot of current cache statistics.
     */
    public CacheStats getStats() {
        return new CacheStats(
                config.getTotalBlocks(),
                getFreeBlockCount(),
                getAllocatedBlockCount(),
                getActiveRequestCount(),
                getUtilization(),
                totalAllocations.get(),
                totalDeallocations.get()
        );
    }

    // ---- Internal ----

    private int blocksRequired(int numTokens) {
        return (int) Math.ceil((double) numTokens / config.getBlockSize());
    }

    private List<Integer> allocateBlocks(int count) {
        if (freeBlocks.size() < count) {
            throw new KVCacheExhaustedException(
                    "KV-Cache pool exhausted. Free: " + freeBlocks.size() +
                    ", Requested: " + count + ". " +
                    "Consider increasing totalBlocks or evicting idle requests.");
        }

        List<Integer> allocated = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            Integer blockId = freeBlocks.pollFirst();
            if (blockId == null) {
                // Shouldn't happen after the size check, but be safe
                // Return already-allocated blocks
                for (int b : allocated) {
                    freeBlocks.addLast(b);
                }
                throw new KVCacheExhaustedException("Race condition during block allocation");
            }
            allocated.add(blockId);
        }
        return allocated;
    }

    private void enforceMaxBlocksPerRequest(int totalBlocks) {
        if (config.getMaxBlocksPerRequest() > 0 && totalBlocks > config.getMaxBlocksPerRequest()) {
            throw new IllegalStateException(
                    "Request exceeds max blocks per request limit: " + totalBlocks +
                    " > " + config.getMaxBlocksPerRequest());
        }
    }

    @Override
    public void close() {
        // Free all remaining allocations
        requestBlocks.clear();
        requestTokenCounts.clear();
        freeBlocks.clear();
        blockPool.close();
    }

    @Override
    public String toString() {
        return "PagedKVCacheManager{" +
                "free=" + getFreeBlockCount() +
                ", allocated=" + getAllocatedBlockCount() +
                ", active=" + getActiveRequestCount() +
                ", utilization=" + String.format("%.1f%%", getUtilization() * 100) +
                '}';
    }

    // ---- Records ----

    /**
     * Snapshot of cache statistics for monitoring.
     */
    public record CacheStats(
            int totalBlocks,
            int freeBlocks,
            int allocatedBlocks,
            int activeRequests,
            double utilization,
            long totalAllocations,
            long totalDeallocations
    ) {}
}
