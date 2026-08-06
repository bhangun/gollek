package tech.kayys.gollek.runtime.distributed;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.Map;

public interface PartitionExecutor {
    Map<String, Tensor> execute(Partition partition, Map<String, Tensor> inputs);
}
