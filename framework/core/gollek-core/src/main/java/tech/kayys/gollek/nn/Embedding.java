package tech.kayys.gollek.nn;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.List;

public final class Embedding extends Module {
    private final Tensor weight;

    public Embedding(Tensor weight) {
        this.weight = weight;
    }

    @Override
    public Tensor forward(Tensor input) {
        // Simple embedding lookup using slice or specialized kernel
        // For now, assuming input contains indices and we use a backend op
        return weight.slice(new long[] { 0, 0 }, new long[] { 1, 1 }); // Placeholder for actual lookup
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(weight);
    }
}
