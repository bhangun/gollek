package tech.kayys.gollek.diffusion.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import java.util.Map;

/**
 * Node for style/visual prompt conditioning.
 */
public final class ImagePromptConditioningNode implements VisualGraphNode {

    private final float styleWeight;

    public ImagePromptConditioningNode(float styleWeight) {
        this.styleWeight = styleWeight;
    }

    public ImagePromptConditioningNode() {
        this(0.8f);
    }

    @Override
    public String id() {
        return "image_prompt_conditioning";
    }

    @Override
    public Tensor compute(Map<String, Tensor> inputs) {
        Tensor textCond = inputs.get("text_cond");
        Tensor imgCond = inputs.get("img_cond");
        if (textCond != null && imgCond != null) {
            return textCond.add(imgCond.mul(styleWeight));
        }
        return textCond;
    }
}
