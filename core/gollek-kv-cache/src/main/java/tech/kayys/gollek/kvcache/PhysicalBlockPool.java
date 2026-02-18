package tech.kayys.gollek.kvcache;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Pre-allocated contiguous memory pool for KV-Cache blocks.
 * <p>
 * Manages a large slab of off-heap memory using the Java FFM API ({@link Arena}).
 * Each block stores Keys and Values for {@code blockSize} tokens across all
 * model layers and heads.
 * <p>
 * In production with GPU support, this would allocate GPU VRAM via LibTorch.
 * In CPU mode (default for development), it uses {@link Arena#ofShared()} for
 * off-heap native memory that is thread-safe and can be passed to native kernels.
 * <p>
 * The pool is a flat contiguous memory region. Individual blocks are accessed
 * by computing byte offsets: {@code blockId × bytesPerBlock}.
 *
 * @see KVCacheConfig
 */
public class PhysicalBlockPool implements AutoCloseable {

    private final Arena arena;
    private final MemorySegment kPoolSlab;
    private final MemorySegment vPoolSlab;
    private final KVCacheConfig config;
    private final long bytesPerBlock;

    /**
     * Allocate the physical block pool.
     *
     * @param config the KV-Cache configuration
     */
    public PhysicalBlockPool(KVCacheConfig config) {
        this.config = config;
        this.arena = Arena.ofShared();

        // Each block stores: blockSize × numHeads × headDim values in Float16 (2 bytes)
        // Note: numLayers is handled by the cache manager which stores separate
        // K/V entries per layer. Here we store a single layer's K or V per block.
        // For a complete implementation, each block stores ALL layers.
        this.bytesPerBlock = (long) config.getBlockSize()
                * config.getNumHeads()
                * config.getHeadDim()
                * 2L; // Float16 = 2 bytes per element

        long totalKBytes = bytesPerBlock * config.getTotalBlocks() * config.getNumLayers();
        long totalVBytes = totalKBytes;

        this.kPoolSlab = arena.allocate(totalKBytes, 64); // 64-byte aligned for SIMD/GPU
        this.vPoolSlab = arena.allocate(totalVBytes, 64);

        // Zero-initialize
        kPoolSlab.fill((byte) 0);
        vPoolSlab.fill((byte) 0);
    }

    /**
     * Get the memory segment for a specific K-cache block at a specific layer.
     *
     * @param blockId the physical block ID
     * @param layer   the transformer layer index
     * @return a memory segment slice for the requested K-cache block
     * @throws IndexOutOfBoundsException if blockId or layer is out of range
     */
    public MemorySegment getKBlock(int blockId, int layer) {
        validateBlockId(blockId);
        validateLayer(layer);
        long offset = computeOffset(blockId, layer);
        return kPoolSlab.asSlice(offset, bytesPerBlock);
    }

    /**
     * Get the memory segment for a specific V-cache block at a specific layer.
     *
     * @param blockId the physical block ID
     * @param layer   the transformer layer index
     * @return a memory segment slice for the requested V-cache block
     * @throws IndexOutOfBoundsException if blockId or layer is out of range
     */
    public MemorySegment getVBlock(int blockId, int layer) {
        validateBlockId(blockId);
        validateLayer(layer);
        long offset = computeOffset(blockId, layer);
        return vPoolSlab.asSlice(offset, bytesPerBlock);
    }

    /**
     * Get the raw K pool slab for passing to native kernels.
     * The kernel is responsible for interpreting the memory layout.
     */
    public MemorySegment rawKPool() {
        return kPoolSlab;
    }

    /**
     * Get the raw V pool slab for passing to native kernels.
     */
    public MemorySegment rawVPool() {
        return vPoolSlab;
    }

    /**
     * Get the block table (physical block IDs) as a native int array segment
     * suitable for passing to native kernels.
     *
     * @param blockIds the list of physical block IDs
     * @return a MemorySegment containing the block IDs as native ints
     */
    public MemorySegment packBlockTable(int[] blockIds) {
        return arena.allocateFrom(ValueLayout.JAVA_INT, blockIds);
    }

    /**
     * Get bytes per block (single layer).
     */
    public long getBytesPerBlock() {
        return bytesPerBlock;
    }

    /**
     * Get total allocated memory in bytes (K + V pools combined).
     */
    public long getTotalAllocatedBytes() {
        long perPool = bytesPerBlock * config.getTotalBlocks() * config.getNumLayers();
        return perPool * 2; // K + V
    }

    /**
     * Get the configuration used to create this pool.
     */
    public KVCacheConfig getConfig() {
        return config;
    }

    private long computeOffset(int blockId, int layer) {
        // Layout: [block0_layer0, block0_layer1, ..., block1_layer0, ...]
        return ((long) blockId * config.getNumLayers() + layer) * bytesPerBlock;
    }

    private void validateBlockId(int blockId) {
        if (blockId < 0 || blockId >= config.getTotalBlocks()) {
            throw new IndexOutOfBoundsException(
                    "Block ID " + blockId + " out of range [0, " + config.getTotalBlocks() + ")");
        }
    }

    private void validateLayer(int layer) {
        if (layer < 0 || layer >= config.getNumLayers()) {
            throw new IndexOutOfBoundsException(
                    "Layer " + layer + " out of range [0, " + config.getNumLayers() + ")");
        }
    }

    @Override
    public void close() {
        arena.close();
    }

    @Override
    public String toString() {
        return "PhysicalBlockPool{" +
                "totalBlocks=" + config.getTotalBlocks() +
                ", bytesPerBlock=" + bytesPerBlock +
                ", totalAllocatedMB=" + (getTotalAllocatedBytes() / (1024 * 1024)) +
                '}';
    }
}
