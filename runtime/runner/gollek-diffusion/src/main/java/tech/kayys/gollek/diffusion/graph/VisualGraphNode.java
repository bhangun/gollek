package tech.kayys.gollek.diffusion.graph;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

/**
 * Agnostic visual generation graph computation node.
 */
public interface VisualGraphNode {
    String id();
    Tensor compute(Map<String, Tensor> inputs);
}
