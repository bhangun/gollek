package tech.kayys.gollek.diffusion.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.TensorFactory;

import java.util.Map;

/**
 * Node for dual text encoding conditioning.
 */
public final class DualTextEncoderNode implements VisualGraphNode {

    private final String prompt;

    public DualTextEncoderNode(String prompt) {
        this.prompt = prompt;
    }

    public String prompt() {
        return prompt;
    }

    @Override
    public String id() {
        return "dual_text_encoder";
    }

    @Override
    public Tensor compute(Map<String, Tensor> inputs) {
        if (inputs.containsKey("clip") && inputs.containsKey("t5")) {
            return inputs.get("clip");
        }
        return TensorFactory.zeros(1L, 512L, 4096L);
    }
}
