package tech.kayys.gollek.core.params;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.*;

/**
 * Usage (NO MORE STRING CONCAT)
 * ParamPath base = ParamPath.root("layers").index(0);
 * String qWeight = base.child("attn").child("q_proj").child("weight").key();
 * Result:
 * layers.0.attn.q_proj.weight
 * Benefits
 * deterministic
 * hierarchical
 * compatible with HuggingFace / PyTorch
 * no typo bugs
 * easy mapping
 */

public final class ParamRegistry {
    private final Map<String, Tensor> params = new HashMap<>();

    public void put(ParamPath path, Tensor t) {
        params.put(path.key(), t);
    }

    public Tensor get(ParamPath path) {
        return params.get(path.key());
    }

    public Map<String, Tensor> all() {
        return params;
    }
}