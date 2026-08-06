package tech.kayys.gollek.runtime.exec;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.kv.*;
import tech.kayys.gollek.runtime.client.RemoteClient;

public final class RemoteExecutionProvider implements ExecutionProvider {
    private final RemoteClient client;

    public RemoteExecutionProvider(RemoteClient client) {
        this.client = client;
    }

    @Override
    public KVCacheSnapshot prefill(
            Tensor prompt,
            Tensor wqkv,
            int heads,
            int maxSeq) {
        return client.prefill(prompt, wqkv, heads, maxSeq);
    }

    @Override
    public Tensor decodeStep(
            Tensor token,
            Tensor wqkv,
            KVCache cache,
            int heads) {
        return client.decode(token, wqkv, cache, heads);
    }
}
