package tech.kayys.gollek.runtime.graph;

import tech.kayys.gollek.runtime.tensor.*;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * A tracked memory block managed by the graph memory planner.
 * <p>
 * Each block wraps a native memory segment and tracks when it becomes
 * free for reuse based on lifetime analysis.
 */
final class MemoryBlock {
    final TensorKey key;
    final MemorySegment segment;
    int freeAt;

    MemoryBlock(TensorKey key, MemorySegment segment) {
        this.key = key;
        this.segment = segment;
        this.freeAt = Integer.MAX_VALUE; // initially in use
    }
}

/**
 * Lifetime-aware memory allocator for graph execution.
 * <p>
 * Uses the lifetime information from {@link LifetimeAnalyzer} to reuse
 * memory blocks whose lifetimes don't overlap. This reduces peak memory
 * from O(n_nodes) to O(max_concurrent_tensors).
 * <p>
 * This is the same technique used by XLA and TensorRT.
 */
public final class GraphMemoryPlanner {

    private final List<MemoryBlock> active = new ArrayList<>();

    /**
     * Allocate (or reuse) a memory segment for the given tensor key.
     *
     * @param key          tensor shape + dtype + device
     * @param currentIndex current execution step index
     * @param backend      backend for fresh allocations
     * @param pool         tensor pool for allocation
     * @param ctx          execution context
     * @return a native memory segment (possibly reused)
     */
    public MemorySegment allocate(
        TensorKey key,
        int currentIndex,
        Backend backend,
        TensorPool pool,
        ExecutionContext ctx
    ) {
        // Try to reuse an expired block with matching key
        for (MemoryBlock block : active) {
            if (block.freeAt <= currentIndex && block.key.equals(key)) {
                block.freeAt = Integer.MAX_VALUE; // mark as in-use again
                return block.segment;
            }
        }

        // Allocate new via pool
        MemorySegment seg = pool.acquire(key, backend, ctx);
        active.add(new MemoryBlock(key, seg));
        return seg;
    }

    /**
     * Release tensors whose lifetime has ended at the current step.
     * Marks their memory blocks as available for reuse.
     *
     * @param currentIndex current execution step index
     * @param nodes        all nodes in the plan
     */
    public void releaseExpired(int currentIndex, List<GraphNode> nodes) {
        for (GraphNode node : nodes) {
            if (node.lastUse == currentIndex && node.output != null) {
                // Mark the corresponding block as free
                for (MemoryBlock block : active) {
                    if (block.segment.equals(node.output.nativeHandle())) {
                        block.freeAt = currentIndex;
                        break;
                    }
                }
            }
        }
    }

    /** Number of active memory blocks. */
    public int activeBlocks() {
        return active.size();
    }
}
