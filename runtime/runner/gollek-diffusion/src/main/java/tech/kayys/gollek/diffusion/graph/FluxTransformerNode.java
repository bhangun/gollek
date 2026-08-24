package tech.kayys.gollek.diffusion.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.TensorFactory;

import java.util.Map;

/**
 * Node for DiT / UNet transformer step evaluation.
 */
public final class FluxTransformerNode implements VisualGraphNode {

    private final float timestep;
    private final float guidanceScale;

    public FluxTransformerNode(float timestep, float guidanceScale) {
        this.timestep = timestep;
        this.guidanceScale = guidanceScale;
    }

    @Override
    public String id() {
        return "flux_transformer_step";
    }

    @Override
    public Tensor compute(Map<String, Tensor> inputs) {
        Tensor latents = inputs.get("latents");
        return latents != null ? latents : TensorFactory.zeros(1L, 16L, 64L, 64L);
    }
}
