package tech.kayys.gollek.diffusion.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import java.util.Map;

/**
 * Node for Flow-Matching Euler solver step.
 */
public final class FlowEulerSchedulerNode implements VisualGraphNode {

    private final float dt;

    public FlowEulerSchedulerNode(float dt) {
        this.dt = dt;
    }

    @Override
    public String id() {
        return "flow_euler_scheduler";
    }

    @Override
    public Tensor compute(Map<String, Tensor> inputs) {
        Tensor latents = inputs.get("latents");
        Tensor velocity = inputs.get("velocity");
        if (latents != null && velocity != null) {
            return latents.add(velocity.mul(dt));
        }
        return latents;
    }
}
