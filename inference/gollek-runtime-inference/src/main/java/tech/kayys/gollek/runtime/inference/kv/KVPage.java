package tech.kayys.gollek.runtime.inference.kv;

import java.lang.foreign.MemorySegment;

/**
 * A fixed-size page of KV cache memory.
 * <p>
 * Pages are the allocation unit for {@link PagedKVCache}, enabling
 * fine-grained memory management without fragmentation.
 * This is the same approach used by vLLM's PagedAttention.
 */
public final class KVPage {

    /** Native memory for key projections. */
    public final MemorySegment k;

    /** Native memory for value projections. */
    public final MemorySegment v;

    /** Maximum number of tokens this page can hold. */
    public final int capacity;

    /** Number of tokens currently stored in this page. */
    public int size;

    public KVPage(MemorySegment k, MemorySegment v, int capacity) {
        this.k = k;
        this.v = v;
        this.capacity = capacity;
        this.size = 0;
    }

    /** Whether this page has room for more tokens. */
    public boolean hasSpace() {
        return size < capacity;
    }

    @Override
    public String toString() {
        return "KVPage[size=" + size + "/" + capacity + "]";
    }
}
