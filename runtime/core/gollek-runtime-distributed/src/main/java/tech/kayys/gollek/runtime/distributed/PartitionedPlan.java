package tech.kayys.gollek.runtime.distributed;

import java.util.List;

public final class PartitionedPlan {
    private final List<Partition> partitions;

    public PartitionedPlan(List<Partition> partitions) {
        this.partitions = partitions;
    }

    public List<Partition> partitions() {
        return partitions;
    }
}
