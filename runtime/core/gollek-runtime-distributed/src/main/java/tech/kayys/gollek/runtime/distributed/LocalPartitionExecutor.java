package tech.kayys.gollek.runtime.distributed;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.ExecutionEngine;
import java.util.Map;

public final class LocalPartitionExecutor implements PartitionExecutor {
    private final ExecutionEngine engine;

    public LocalPartitionExecutor(ExecutionEngine engine) {
        this.engine = engine;
    }

    @Override
    public Map<String, Tensor> execute(
            Partition partition,
            Map<String, Tensor> inputs) {
        // Assuming ExecutionEngine has a way to run a partition or sub-graph
        // For now, we'll just throw an exception since ExecutionEngine.runPartition doesn't exist
        throw new UnsupportedOperationException("Partition execution not implemented in ExecutionEngine yet");
    }
}