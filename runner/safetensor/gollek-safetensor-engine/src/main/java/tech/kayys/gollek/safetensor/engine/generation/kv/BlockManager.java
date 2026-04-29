package tech.kayys.gollek.safetensor.engine.generation.kv;

import jakarta.enterprise.context.ApplicationScoped;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Manages physical memory blocks for KV cache.
 * Refactored to use contiguous memory slabs for Metal compatibility.
 * 
 * Layout in slab: [maxBlocks, numHeads, tokensPerBlock, headDim]
 * This matches the expectation of the native Metal/MPS attention kernels.
 */
@ApplicationScoped
public class BlockManager {
    private MemorySegment kPoolSlab;
    private MemorySegment vPoolSlab;
    private int blockSizeBytes = 0;
    private int tokensPerBlock = 0;
    private int numHeads = 0;
    private int headDim = 0;
    private long headStride = 0;
    private long tokenStride = 0;
    private boolean initialized = false;

    private final Arena poolArena = Arena.ofAuto();
    private final ConcurrentLinkedQueue<Integer> freeBlockIndices = new ConcurrentLinkedQueue<>();
    private final AtomicInteger totalAllocatedBlocks = new AtomicInteger(0);

    public void initialize(int tokensPerBlock, int numHeads, int headDim, int maxBlocks) {
        if (initialized)
            return;

        this.tokensPerBlock = tokensPerBlock;
        this.numHeads = numHeads;
        this.headDim = headDim;

        // Layout: [head, token, dim] per block
        this.tokenStride = headDim;
        this.headStride = (long) tokensPerBlock * tokenStride;
        this.blockSizeBytes = (int) (numHeads * headStride * 4);

        long totalBytes = (long) maxBlocks * blockSizeBytes;

        // Allocate contiguous slabs for K and V
        this.kPoolSlab = poolArena.allocate(totalBytes, 64);
        this.vPoolSlab = poolArena.allocate(totalBytes, 64);

        for (int i = 0; i < maxBlocks; i++) {
            freeBlockIndices.add(i);
        }

        this.initialized = true;
    }

    public Integer allocateBlock() {
        Integer index = freeBlockIndices.poll();
        if (index != null) {
            totalAllocatedBlocks.incrementAndGet();
        }
        return index;
    }

    public void freeBlock(int index) {
        freeBlockIndices.add(index);
        totalAllocatedBlocks.decrementAndGet();
    }

    public MemorySegment getKBlock(int blockIndex) {
        return kPoolSlab.asSlice((long) blockIndex * blockSizeBytes, blockSizeBytes);
    }

    public MemorySegment getVBlock(int blockIndex) {
        return vPoolSlab.asSlice((long) blockIndex * blockSizeBytes, blockSizeBytes);
    }

    public MemorySegment getRawKPool() {
        return kPoolSlab;
    }

    public MemorySegment getRawVPool() {
        return vPoolSlab;
    }

    public int getBlockSizeBytes() {
        return blockSizeBytes;
    }

    public int getTokensPerBlock() {
        return tokensPerBlock;
    }

    public long getHeadStride() {
        return headStride;
    }

    public long getTokenStride() {
        return tokenStride;
    }
}
