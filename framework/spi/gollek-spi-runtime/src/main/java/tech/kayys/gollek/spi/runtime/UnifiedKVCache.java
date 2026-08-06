/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.Collection;

/**
 * High-level abstraction for a complete model session's KV memory.
 * Bridges the gap between different runner-specific KV implementations.
 */
public interface UnifiedKVCache extends AutoCloseable {

    /**
     * The model ID this cache belongs to.
     */
    String modelId();

    /**
     * Total sequence length stored in this cache.
     */
    int seqLength();

    /**
     * Number of layers in the model.
     */
    int numLayers();

    /**
     * Gets the KV block for a specific layer.
     */
    KVBlock getLayer(int layerId);

    /**
     * Gets all layers in this cache.
     */
    Collection<KVBlock> getAllLayers();

    /**
     * Gets the GKV binary header representing the layout of this cache.
     */
    GKVHeader getHeader();

    /**
     * Accesses the raw off-heap memory segment backing this cache.
     */
    MemorySegment asBinarySegment();

    /**
     * Creates a shallow copy (fork) of this cache for sharing between tenants.
     * Uses Copy-on-Write logic if the segment is modified.
     */
    UnifiedKVCache fork();

    /**
     * Creates a specialized fork for speculative decoding.
     * Allows a draft model to append tokens to a temporary block.
     * 
     * @param lookahead the number of tokens expected to be drafted
     * @return a forked cache that isolates speculative writes
     */
    UnifiedKVCache forkForSpeculation(int lookahead);

    /**
     * Serializes the cache into a portable ByteBuffer for paging or network.
     */
    ByteBuffer serialize();

    /**
     * Disposes of the underlying memory segments once the cache is no longer needed.
     */
    @Override
    void close();
}
