package tech.kayys.gollek.runtime.exec;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.kv.*;

/**
 * Current pipeline:
 * PrefillRunner + KVCache + DecodeRunner
 * 
 * Adaptive layer:
 * AdaptiveExecutionProvider (adaptive)
 * ├── LocalExecution
 * └── RemoteExecution
 * 
 * Switch at runtime, not in model code.
 * 
 * Modes
 * LOCAL_ONLY → everything in one process
 * HYBRID → prefill local, decode remote OR vice versa
 * REMOTE_ONLY → everything remote
 * 
 * Key abstraction:
 * ExecutionProvider decides:
 * - where prefill runs
 * - where decode runs
 * - how KV moves
 */
public interface ExecutionProvider {
    KVCacheSnapshot prefill(
            Tensor prompt,
            Tensor wqkv,
            int heads,
            int maxSeq);

    Tensor decodeStep(
            Tensor token,
            Tensor wqkv,
            KVCache cache,
            int heads);
}