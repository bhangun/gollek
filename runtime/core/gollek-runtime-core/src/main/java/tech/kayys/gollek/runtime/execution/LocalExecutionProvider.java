package tech.kayys.gollek.runtime.exec;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.kv.*;
import tech.kayys.gollek.runtime.service.*;

public final class LocalExecutionProvider implements ExecutionProvider {
    private final PrefillService prefill;
    private final DecodeService decode;

    public LocalExecutionProvider(KVCodec codec) {
        this.prefill = new PrefillService(codec);
        this.decode = new DecodeService(codec);
    }

    @Override
    public KVCacheSnapshot prefill(
            Tensor prompt,
            Tensor wqkv,
            int heads,
            int maxSeq) {
        return prefill.prefill(prompt, wqkv, heads, maxSeq);
    }

    @Override
    public Tensor decodeStep(
            Tensor token,
            Tensor wqkv,
            KVCache cache,
            int heads) {
        return decode.step(token, wqkv, cache, heads);
    }
}