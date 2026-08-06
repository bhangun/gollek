package tech.kayys.gollek.runtime.pipeline;

import tech.kayys.gollek.runtime.distributed.Partition;

public final class PipelineStage {
    public final Partition partition;

    public PipelineStage(Partition partition) {
        this.partition = partition;
    }
}
