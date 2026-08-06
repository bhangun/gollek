package tech.kayys.gollek.runtime.distributed;

import tech.kayys.gollek.core.graph.Node;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import java.util.List;

public final class Partition {
    private final String id;
    private final DeviceType device;
    private final List<Node> steps;

    public Partition(String id, DeviceType device, List<Node> steps) {
        this.id = id;
        this.device = device;
        this.steps = steps;
    }

    public String id() {
        return id;
    }

    public DeviceType device() {
        return device;
    }

    public List<Node> steps() {
        return steps;
    }
}
