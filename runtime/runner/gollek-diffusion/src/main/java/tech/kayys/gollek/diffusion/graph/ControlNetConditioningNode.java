package tech.kayys.gollek.diffusion.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import java.util.Map;

/**
 * Node for spatial constraint conditioning.
 */
public final class ControlNetConditioningNode implements VisualGraphNode {

    private final float conditioningScale;

    public ControlNetConditioningNode(float conditioningScale) {
        this.conditioningScale = conditioningScale;
    }

    public ControlNetConditioningNode() {
        this(1.0f);
    }

    @Override
    public String id() {
        return "controlnet_conditioning";
    }

    @Override
    public Tensor compute(Map<String, Tensor> inputs) {
        Tensor latents = inputs.get("latents");
        Tensor hint = inputs.get("hint");
        if (latents != null && hint != null) {
            return latents.add(hint.mul(conditioningScale));
        }
        return latents;
    }
}
