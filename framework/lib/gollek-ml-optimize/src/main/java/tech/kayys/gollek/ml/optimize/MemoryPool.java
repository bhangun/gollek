package tech.kayys.gollek.ml.optimize;

import tech.kayys.gollek.ml.tensor.RunnerDevice;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Shared memory pool that provides efficient tensor allocation across all
 * frameworks (LibTorch, LiteRT, ONNX, GGUF, etc.) with zero-copy support.
 *
 * <h2>Problem</h2>
 * <p>Each framework manages its own memory. Transferring tensors between
 * frameworks requires expensive copies.
 *
 * <h2>Solution</h2>
 * <p>A unified memory pool that all frameworks share:
 * <pre>
 *   MemoryPool pool = MemoryPool.builder()
 *       .device(RunnerDevice.CUDA)
 *       .maxSize(16L * 1024 * 1024 * 1024)  // 16GB
 *       .strategy(MemoryStrategy.FRAGMENT_AWARE)
 *       .build();
 *
 *   // Allocate from pool (zero-copy between frameworks)
 *   MemorySegment tensor = pool.allocate(1024 * 1024);
 *   libtorchRunner.infer(tensor);
 *   onnxRunner.infer(tensor);  // No copy needed!
 * </pre>
 *
 * @since 0.3.0
 */
public class MemoryPool {

    private final RunnerDevice device;
    private final long maxSize;
    private final MemoryStrategy strategy;
    private final Arena arena;

    private final AtomicLong allocatedBytes = new AtomicLong(0);
    private final AtomicLong peakBytes = new AtomicLong(0);
    private final AtomicLong allocationCount = new AtomicLong(0);
    private final AtomicLong freeCount = new AtomicLong(0);
    private final Map<Long, Long> allocations = new ConcurrentHashMap<>();

    private MemoryPool(Builder builder) {
        this.device = builder.device;
        this.maxSize = builder.maxSize;
        this.strategy = builder.strategy;
        this.arena = switch (device) {
            case CPU, AUTO -> Arena.ofAuto();
            default -> Arena.ofConfined();  // GPU would use native arena
        };
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Allocates memory from the pool.
     *
     * @param bytes number of bytes to allocate
     * @return memory segment
     * @throws OutOfMemoryError if pool is exhausted
     */
    public MemorySegment allocate(long bytes) {
        long current = allocatedBytes.getAndUpdate(a -> a + bytes);
        if (current + bytes > maxSize) {
            allocatedBytes.updateAndGet(a -> a - bytes);  // Roll back
            throw new OutOfMemoryError(String.format(
                "MemoryPool exhausted: requested %d, available %d/%d",
                bytes, maxSize - current, maxSize));
        }

        MemorySegment segment = arena.allocate(bytes);
        allocations.put(segment.address(), bytes);
        allocationCount.incrementAndGet();
        peakBytes.updateAndGet(p -> Math.max(p, allocatedBytes.get()));

        return segment;
    }

    /**
     * Allocates aligned memory (useful for SIMD operations).
     *
     * @param bytes number of bytes
     * @param alignment alignment boundary (e.g., 64 for AVX-512)
     */
    public MemorySegment allocateAligned(long bytes, int alignment) {
        // Over-allocate and align
        long padded = bytes + alignment;
        MemorySegment raw = allocate(padded);
        long address = raw.address();
        long alignedAddress = (address + alignment - 1) & ~(alignment - 1);
        long offset = alignedAddress - address;
        return raw.asSlice(offset, bytes);
    }

    /**
     * Frees memory back to the pool.
     *
     * @param segment the segment to free
     */
    public void free(MemorySegment segment) {
        Long size = allocations.remove(segment.address());
        if (size != null) {
            allocatedBytes.updateAndGet(a -> a - size);
            freeCount.incrementAndGet();
        }
    }

    /**
     * Gets pool utilization as a percentage.
     */
    public double utilizationPercent() {
        return (double) allocatedBytes.get() / maxSize * 100.0;
    }

    /**
     * Gets available memory in bytes.
     */
    public long availableMemory() {
        return maxSize - allocatedBytes.get();
    }

    /**
     * Gets pool statistics.
     */
    public PoolStats stats() {
        return new PoolStats(
            maxSize,
            allocatedBytes.get(),
            peakBytes.get(),
            allocationCount.get(),
            freeCount.get(),
            allocations.size()
        );
    }

    /**
     * Resets peak memory counter.
     */
    public void resetPeak() {
        peakBytes.set(allocatedBytes.get());
    }

    /**
     * Closes the pool and releases all memory.
     */
    public void close() {
        arena.close();
    }

    /**
     * Memory allocation strategy.
     */
    public enum MemoryStrategy {
        /** Best-effort allocation, may fragment */
        BEST_EFFORT,
        /** Defragmentation-aware allocation */
        FRAGMENT_AWARE,
        /** Buddy allocator for reduced fragmentation */
        BUDDY
    }

    /**
     * Pool statistics record.
     */
    public record PoolStats(
        long maxSize,
        long allocatedBytes,
        long peakBytes,
        long allocationCount,
        long freeCount,
        int activeAllocations
    ) {
        public double utilizationPercent() {
            return (double) allocatedBytes / maxSize * 100.0;
        }

        public String formattedAllocated() {
            return formatBytes(allocatedBytes);
        }

        public String formattedMax() {
            return formatBytes(maxSize);
        }

        private static String formatBytes(long bytes) {
            if (bytes < 1024) return bytes + " B";
            if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
            if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024.0));
            return String.format("%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
        }
    }

    /**
     * Builder for MemoryPool.
     */
    public static class Builder {
        private RunnerDevice device = RunnerDevice.AUTO;
        private long maxSize = 4L * 1024 * 1024 * 1024;  // 4GB default
        private MemoryStrategy strategy = MemoryStrategy.FRAGMENT_AWARE;

        Builder() {}

        public Builder device(RunnerDevice device) { this.device = device; return this; }
        public Builder maxSize(long maxSize) { this.maxSize = maxSize; return this; }
        public Builder maxSizeGB(long gb) { this.maxSize = gb * 1024 * 1024 * 1024; return this; }
        public Builder strategy(MemoryStrategy strategy) { this.strategy = strategy; return this; }

        public MemoryPool build() {
            return new MemoryPool(this);
        }
    }
}
