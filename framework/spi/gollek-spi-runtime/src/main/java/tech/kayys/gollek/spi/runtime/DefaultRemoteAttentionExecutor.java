/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.CompletableFuture;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * Default local-only implementation of the RemoteAttentionExecutor.
 * In a standard local deployment, this implementation assumes all attention
 * is computed locally and throws an exception if remote execution is explicitly required.
 */
@ApplicationScoped
public class DefaultRemoteAttentionExecutor implements RemoteAttentionExecutor {

    @Override
    public CompletableFuture<MemorySegment> executeRemote(
        KVKey key, 
        MemorySegment query, 
        KVLocation location, 
        AttentionKernel.AttentionContext context
    ) {
        // By default, we don't support remote execution unless a specialized plugin is loaded.
        return CompletableFuture.failedFuture(
            new UnsupportedOperationException("Remote attention execution not supported in local-only mode. Node: " + location.nodeId())
        );
    }

    @Override
    public String protocol() {
        return "local";
    }
}
