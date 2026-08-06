package tech.kayys.gollek.runtime.exec;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.kv.*;

public interface RemoteClientExec {
    KVCacheSnapshot prefill(
            Tensor prompt,
            Tensor wqkv,
            int heads,
            int maxSeq);

    Tensor decode(
            Tensor token,
            Tensor wqkv,
            KVCache cache,
            int heads);
}