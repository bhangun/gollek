package tech.kayys.gollek.sdk.core;

/**
 * Memory management utilities for tensor operations.
 *
 * <p>Provides memory pool, caching allocator, and memory tracking for efficient
 * tensor memory management.</p>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public final class MemoryManager {

    private static volatile MemoryManager instance;

    private long allocatedBytes = 0;
    private long peakBytes = 0;
    private long totalAllocations = 0;
    private long totalFrees = 0;

    private MemoryManager() {}

    /**
     * Get singleton instance.
     */
    public static MemoryManager getInstance() {
        if (instance == null) {
            synchronized (MemoryManager.class) {
                if (instance == null) {
                    instance = new MemoryManager();
                }
            }
        }
        return instance;
    }

    /**
     * Track memory allocation.
     *
     * @param bytes number of bytes allocated
     */
    public synchronized void allocate(long bytes) {
        allocatedBytes += bytes;
        peakBytes = Math.max(peakBytes, allocatedBytes);
        totalAllocations++;
    }

    /**
     * Track memory free.
     *
     * @param bytes number of bytes freed
     */
    public synchronized void free(long bytes) {
        allocatedBytes -= bytes;
        totalFrees++;
    }

    /**
     * Get currently allocated memory.
     */
    public synchronized long getAllocatedBytes() {
        return allocatedBytes;
    }

    /**
     * Get peak memory usage.
     */
    public synchronized long getPeakBytes() {
        return peakBytes;
    }

    /**
     * Get total number of allocations.
     */
    public synchronized long getTotalAllocations() {
        return totalAllocations;
    }

    /**
     * Get total number of frees.
     */
    public synchronized long getTotalFrees() {
        return totalFrees;
    }

    /**
     * Reset statistics.
     */
    public synchronized void reset() {
        allocatedBytes = 0;
        peakBytes = 0;
        totalAllocations = 0;
        totalFrees = 0;
    }

    /**
     * Get memory statistics as a string.
     */
    public synchronized String getStats() {
        return String.format(
            "MemoryStats{allocated=%s, peak=%s, allocations=%d, frees=%d}",
            formatBytes(allocatedBytes),
            formatBytes(peakBytes),
            totalAllocations,
            totalFrees
        );
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.2f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
        return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
}
