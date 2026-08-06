/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.CompletableFuture;

/**
 * Interface for RPC-based remote attention execution.
 * Instead of moving massive KV shards across the network, this executor
 * sends the lightweight Query vector to the node holding the KV shard,
 * returning only the final logits/output.
 */
public interface RemoteAttentionExecutor {

    /**
     * Executes the attention operation on a remote node.
     *
     * @param key      the KV shard identifier
     * @param query    the query tensor (KB size)
     * @param location the remote node location
     * @param context  additional attention context (scaling, seq pos)
     * @return a future resolving to the output tensor segment
     */
    CompletableFuture<MemorySegment> executeRemote(
        KVKey key, 
        MemorySegment query, 
        KVLocation location, 
        AttentionKernel.AttentionContext context
    );

    /**
     * The transport protocol used (e.g. "gRPC", "RDMA", "HTTP").
     */
    String protocol();
}
