/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Interface implemented by runners to bridge their internal memory
 * with the Unified KV representation.
 */
public interface KVAdapter {

    /**
     * Exports the runner's internal KV memory into a Unified cache.
     *
     * @param context the session-specific runner context
     * @return a memory-backed UnifiedKVCache
     */
    UnifiedKVCache exportKV(RuntimeSession context);

    /**
     * Imports a Unified cache into the runner's internal memory.
     *
     * @param kv      the unified cache to import
     * @param context the target session's context
     */
    void importKV(UnifiedKVCache kv, RuntimeSession context);

    /**
     * Whether this adapter can perform zero-copy (shared memory)
     * imports/exports.
     */
    boolean supportsZeroCopy();
}
