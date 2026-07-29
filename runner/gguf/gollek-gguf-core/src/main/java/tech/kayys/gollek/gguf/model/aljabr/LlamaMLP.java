package tech.kayys.gollek.gguf.model.alkhawarizm;

import tech.kayys.alkhawarizm.core.nn.Linear;
import tech.kayys.alkhawarizm.core.nn.Module;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.WeightAdapter;

public class LlamaMLP extends Module {
    private final Linear gateProj;
    private final Linear upProj;
    private final Linear downProj;

    public LlamaMLP(WeightAdapter weights, int layerIdx) {
        Tensor wGate = weights.getWeight("blk." + layerIdx + ".ffn_gate.weight");
        Tensor wUp = weights.getWeight("blk." + layerIdx + ".ffn_up.weight");
        Tensor wDown = weights.getWeight("blk." + layerIdx + ".ffn_down.weight");

        gateProj = new Linear(wGate, null);
        upProj = new Linear(wUp, null);
        downProj = new Linear(wDown, null);

        registerModule("gate_proj", gateProj);
        registerModule("up_proj", upProj);
        registerModule("down_proj", downProj);
    }

    @Override
    public Tensor forward(Tensor x) {
        // SwiGLU: (x W_gate * sigmoid(x W_gate)) * (x W_up) -> W_down
        Tensor gate = gateProj.forward(x);
        Tensor up = upProj.forward(x);

        // silu(x) = x * sigmoid(x)
        Tensor silu = gate.silu();
        Tensor hidden = silu.mul(up);

        return downProj.forward(hidden);
    }
}
