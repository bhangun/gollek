package tech.kayys.gollek.diffusion.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.TensorFactory;

import java.util.Map;

/**
 * Node for inpainting mask-guided latent blending.
 */
public final class InpaintingBlendNode implements VisualGraphNode {

    @Override
    public String id() {
        return "inpainting_blend";
    }

    @Override
    public Tensor compute(Map<String, Tensor> inputs) {
        Tensor orig = inputs.get("original");
        Tensor denoised = inputs.get("denoised");
        Tensor mask = inputs.get("mask");

        if (orig != null && denoised != null && mask != null) {
            Tensor ones = TensorFactory.ones(mask.shape().dims());
            Tensor unmaskedPart = orig.mul(ones.sub(mask));
            Tensor maskedPart = denoised.mul(mask);
            return unmaskedPart.add(maskedPart);
        }
        return denoised;
    }
}
