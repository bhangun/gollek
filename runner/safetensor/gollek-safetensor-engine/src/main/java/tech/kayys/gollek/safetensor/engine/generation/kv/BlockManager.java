package tech.kayys.gollek.safetensor.engine.generation.kv;

import jakarta.enterprise.context.ApplicationScoped;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import org.jboss.logging.Logger;

/**
 * Manages a global pool of memory blocks for Paged Attention.
 * Blocks are recycled across sessions to minimize GC pressure and fragmentation.
 */
@ApplicationScoped
public class BlockManager implements AutoCloseable {

    private static final Logger log = Logger.getLogger(BlockManager.class);

    private final Arena poolArena = Arena.ofAuto();
    private final ConcurrentLinkedQueue<Integer> freeBlockIndices = new ConcurrentLinkedQueue<>();
    private final List<MemorySegment> blocks = new ArrayList<>();
    private final AtomicInteger totalAllocated = new AtomicInteger(0);

    private int blockSizeBytes = 0;
    private boolean initialized = false;

    /**
     * Initialize the block manager with specific dimensions.
     * 
     * @param tokensPerBlock number of tokens per block (e.g. 16)
     * @param numHeads number of KV heads
     * @param headDim dimension per head
     * @param maxBlocks total blocks to pre-allocate or allow
     */
    public synchronized void init(int tokensPerBlock, int numHeads, int headDim, int maxBlocks) {
        if (initialized) return;
        
        // One block stores [blockSize, numHeads, headDim] for both K and V
        // Actually, we store K and V in separate segments or back-to-back.
        // Let's store them back-to-back: [blockSize, numHeads, headDim, 2]
        this.blockSizeBytes = tokensPerBlock * numHeads * headDim * 2 * (int)ValueLayout.JAVA_FLOAT.byteSize();
        
        log.infof("Initializing BlockManager: %d blocks of %d KB each (Total: %.2f MB)", 
                maxBlocks, blockSizeBytes / 1024, (long)maxBlocks * blockSizeBytes / (1024.0 * 1024.0));

        for (int i = 0; i < maxBlocks; i++) {
            blocks.add(poolArena.allocate(blockSizeBytes, 64)); // 64-byte alignment
            freeBlockIndices.add(i);
        }
        
        totalAllocated.set(maxBlocks);
        this.initialized = true;
    }

    /**
     * Allocate a block index.
     * @return block index or -1 if full
     */
    public int allocateBlock() {
        Integer idx = freeBlockIndices.poll();
        return (idx != null) ? idx : -1;
    }

    /**
     * Return a block to the pool.
     */
    public void freeBlock(int index) {
        if (index >= 0 && index < blocks.size()) {
            freeBlockIndices.add(index);
        }
    }

    /**
     * Get the segment for a physical block index.
     */
    public MemorySegment getBlock(int index) {
        return blocks.get(index);
    }

    public int getFreeCount() {
        return freeBlockIndices.size();
    }

    @Override
    public void close() {
        poolArena.close();
        blocks.clear();
        freeBlockIndices.clear();
    }
}
