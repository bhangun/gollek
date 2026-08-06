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
    public enum KvStorageType {
        FP32,
        INT8,
        INT4
    }

    private MemorySegment kPoolSlab;
    private MemorySegment vPoolSlab;
    private MemorySegment kScaleSlab;
    private MemorySegment vScaleSlab;
    private int blockSizeBytes = 0;
    private int tokensPerBlock = 0;
    private int numHeads = 0;
    private int headDim = 0;
    private long headStride = 0;
    private long tokenStride = 0;
    private long scaleStride = 0;
    private boolean initialized = false;
    private int maxBlocks = 0;
    private KvStorageType storageType = KvStorageType.FP32;
    private int bytesPerElement = Float.BYTES;

    private Arena poolArena = Arena.ofShared();
    private final ConcurrentLinkedQueue<Integer> freeBlockIndices = new ConcurrentLinkedQueue<>();
    private final AtomicInteger totalAllocatedBlocks = new AtomicInteger(0);

    public synchronized void initialize(int tokensPerBlock, int numHeads, int headDim, int maxBlocks) {
        initialize(tokensPerBlock, numHeads, headDim, maxBlocks, KvStorageType.FP32);
    }

    public synchronized void initialize(int tokensPerBlock, int numHeads, int headDim, int maxBlocks, KvStorageType storageType) {
        boolean shapeChanged = this.tokensPerBlock != tokensPerBlock
                || this.numHeads != numHeads
                || this.headDim != headDim
                || this.storageType != storageType;
        boolean capacityInsufficient = this.maxBlocks < maxBlocks;

        if (initialized && !shapeChanged && !capacityInsufficient) {
            resetFreeList();
            return;
        }

        if (initialized) {
            try {
                poolArena.close();
            } catch (Exception ignored) {
            }
            poolArena = Arena.ofShared();
        }

        this.tokensPerBlock = tokensPerBlock;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.maxBlocks = maxBlocks;
        this.storageType = storageType;
        this.bytesPerElement = storageType == KvStorageType.FP32 ? Float.BYTES : Byte.BYTES;

        // Layout: [head, token, dim] per block
        this.tokenStride = headDim;
        this.headStride = (long) tokensPerBlock * tokenStride;
        this.scaleStride = (long) tokensPerBlock;
        long blockElements = (long) numHeads * headStride;
        this.blockSizeBytes = switch (storageType) {
            case FP32 -> Math.toIntExact(blockElements * Float.BYTES);
            case INT8 -> Math.toIntExact(blockElements);
            case INT4 -> Math.toIntExact((blockElements + 1L) / 2L);
        };

        long totalBytes = (long) maxBlocks * blockSizeBytes;

        // Allocate contiguous slabs for K and V
        this.kPoolSlab = poolArena.allocate(totalBytes, 64);
        this.vPoolSlab = poolArena.allocate(totalBytes, 64);
        if (storageType != KvStorageType.FP32) {
            long totalScaleBytes = (long) maxBlocks * numHeads * tokensPerBlock * Float.BYTES;
            this.kScaleSlab = poolArena.allocate(totalScaleBytes, 64);
            this.vScaleSlab = poolArena.allocate(totalScaleBytes, 64);
        } else {
            this.kScaleSlab = null;
            this.vScaleSlab = null;
        }

        resetFreeList();

        this.initialized = true;
    }

    private void resetFreeList() {
        freeBlockIndices.clear();
        totalAllocatedBlocks.set(0);
        for (int i = 0; i < maxBlocks; i++) {
            freeBlockIndices.add(i);
        }
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
        if (blockIndex < 0) {
            throw new IllegalStateException("Invalid KV block index for K cache: " + blockIndex);
        }
        return kPoolSlab.asSlice((long) blockIndex * blockSizeBytes, blockSizeBytes);
    }

    public MemorySegment getVBlock(int blockIndex) {
        if (blockIndex < 0) {
            throw new IllegalStateException("Invalid KV block index for V cache: " + blockIndex);
        }
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

    public long getScaleStride() {
        return scaleStride;
    }

    public KvStorageType getStorageType() {
        return storageType;
    }

    public boolean isInt8Quantized() {
        return storageType == KvStorageType.INT8;
    }

    public boolean isInt4Quantized() {
        return storageType == KvStorageType.INT4;
    }

    public boolean isQuantized() {
        return storageType != KvStorageType.FP32;
    }

    public int getBytesPerElement() {
        return bytesPerElement;
    }

    public MemorySegment getKScaleBlock(int blockIndex) {
        if (kScaleSlab == null) {
            return null;
        }
        long scaleBlockBytes = (long) numHeads * tokensPerBlock * Float.BYTES;
        return kScaleSlab.asSlice(blockIndex * scaleBlockBytes, scaleBlockBytes);
    }

    public MemorySegment getVScaleBlock(int blockIndex) {
        if (vScaleSlab == null) {
            return null;
        }
        long scaleBlockBytes = (long) numHeads * tokensPerBlock * Float.BYTES;
        return vScaleSlab.asSlice(blockIndex * scaleBlockBytes, scaleBlockBytes);
    }

    public synchronized void close() {
        if (!initialized) {
            return;
        }
        try {
            poolArena.close();
        } catch (Exception ignored) {
        }
        poolArena = Arena.ofShared();
        kPoolSlab = null;
        vPoolSlab = null;
        kScaleSlab = null;
        vScaleSlab = null;
        freeBlockIndices.clear();
        totalAllocatedBlocks.set(0);
        initialized = false;
        maxBlocks = 0;
    }
}
