package tech.kayys.gollek.runtime.client;

import tech.kayys.gollek.runtime.plan.ExecutionPlan;
import tech.kayys.gollek.runtime.control.ExecutionSession;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.kv.KVCache;
import tech.kayys.gollek.runtime.kv.KVCacheSnapshot;
import java.util.Map;

public interface RemoteClient {
    Map<String, Tensor> execute(
            ExecutionPlan plan,
            Map<String, Tensor> inputs,
            ExecutionSession session);

    // Added methods needed by RemoteExecutionProvider
    KVCacheSnapshot prefill(Tensor prompt, Tensor wqkv, int heads, int maxSeq);
    Tensor decode(Tensor token, Tensor wqkv, KVCache cache, int heads);
    
    // Added method needed by RemotePartitionExecutor
    Map<String, Tensor> executePartition(tech.kayys.gollek.runtime.distributed.Partition partition, Map<String, Tensor> inputs);
}
