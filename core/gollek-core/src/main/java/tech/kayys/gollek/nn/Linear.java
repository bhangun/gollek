package tech.kayys.gollek.nn;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.List;

public final class Linear extends Module {
    private final Tensor weight;
    private final Tensor bias;

    public Linear(Tensor weight, Tensor bias) {
        this.weight = weight;
        this.bias = bias;
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor out = input.matmul(weight);
        if (bias != null) {
            out = out.add(bias);
        }
        return out;
    }

    @Override
    public List<Tensor> parameters() {
        if (bias != null) {
            return List.of(weight, bias);
        }
        return List.of(weight);
    }
}
