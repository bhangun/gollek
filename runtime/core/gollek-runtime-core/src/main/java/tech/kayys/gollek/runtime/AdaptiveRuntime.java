package tech.kayys.gollek.runtime;


import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.AdaptiveRuntime.Mode;
import tech.kayys.gollek.runtime.exec.*;
import tech.kayys.gollek.runtime.kv.*;

/**
 * 
 * USAGE (SINGLE MACHINE → MULTI MACHINE)
 * 
 * Default (single machine):
 * KVCodec codec = new FP16Codec();
 * ExecutionProvider local = new LocalExecutionProvider(codec);
 * AdaptiveRuntime runtime = new AdaptiveRuntime(
 * local,
 * null,
 * AdaptiveRuntime.Mode.LOCAL_ONLY
 * );
 * 
 * Hybrid (real-world)
 * ExecutionProvider local = new LocalExecutionProvider(codec);
 * ExecutionProvider remote = new RemoteExecutionProvider(new GrpcClient());
 * AdaptiveRuntime runtime = new AdaptiveRuntime(
 * local,
 * remote,
 * AdaptiveRuntime.Mode.HYBRID
 * );
 * 
 * Flow stays IDENTICAL
 * KVCacheSnapshot snap = runtime.prefill(prompt, wqkv, heads, maxSeq);
 * KVCache cache = KVCacheSerializer.restore(snap, codec);
 * Tensor out = runtime.decodeStep(token, wqkv, cache, heads);
 */
public final class AdaptiveRuntime {
    public enum Mode {
        LOCAL_ONLY,
        REMOTE_ONLY,
        HYBRID
    }

    private final ExecutionProvider local;
    private final ExecutionProvider remote;
    private final Mode mode;

    public AdaptiveRuntime(
            ExecutionProvider local,
            ExecutionProvider remote,
            Mode mode) {
        this.local = local;
        this.remote = remote;
        this.mode = mode;
    }

    public KVCacheSnapshot prefill(
            Tensor prompt,
            Tensor wqkv,
            int heads,
            int maxSeq) {
        switch (mode) {
            case LOCAL_ONLY:
                return local.prefill(prompt, wqkv, heads, maxSeq);
            case REMOTE_ONLY:
                return remote.prefill(prompt, wqkv, heads, maxSeq);
            case HYBRID:
                // typical: prefill remote (GPU), decode local (CPU)
                return remote.prefill(prompt, wqkv, heads, maxSeq);
            default:
                throw new InferenceException(ErrorCode.RUNTIME_INVALID_STATE, "Invalid state");
        }
    }

    public Tensor decodeStep(
            Tensor token,
            Tensor wqkv,
            KVCache cache,
            int heads) {
        switch (mode) {
            case LOCAL_ONLY:
                return local.decodeStep(token, wqkv, cache, heads);
            case REMOTE_ONLY:
                return remote.decodeStep(token, wqkv, cache, heads);
            case HYBRID:
                return local.decodeStep(token, wqkv, cache, heads);
            default:
                throw new InferenceException(ErrorCode.RUNTIME_INVALID_STATE, "Invalid state");
        }
    }
}