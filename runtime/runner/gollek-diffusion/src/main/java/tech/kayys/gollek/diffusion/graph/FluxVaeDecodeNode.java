package tech.kayys.gollek.diffusion.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.TensorFactory;

import java.util.Map;

/**
 * Node for 16-channel VAE latent decoding.
 */
public final class FluxVaeDecodeNode implements VisualGraphNode {

    private final float scaleFactor;
    private final float shiftFactor;

    public FluxVaeDecodeNode(float scaleFactor, float shiftFactor) {
        this.scaleFactor = scaleFactor;
        this.shiftFactor = shiftFactor;
    }

    public FluxVaeDecodeNode() {
        this(0.3611f, 0.1159f);
    }

    @Override
    public String id() {
        return "flux_vae_decode";
    }

    @Override
    public Tensor compute(Map<String, Tensor> inputs) {
        Tensor latents = inputs.get("latents");
        return latents != null ? latents : TensorFactory.zeros(1L, 3L, 512L, 512L);
    }
}
