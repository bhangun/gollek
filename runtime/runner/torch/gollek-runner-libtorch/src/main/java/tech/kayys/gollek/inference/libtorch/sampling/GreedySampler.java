package tech.kayys.gollek.inference.libtorch.sampling;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;

/**
 * Greedy (argmax) sampling — always selects the highest-probability token.
 * <p>
 * Deterministic and fastest, but produces repetitive output.
 */
public class GreedySampler implements SamplingStrategy {

    @Override
    public long sample(TorchTensor logits) {
        try (TorchTensor argmax = logits.argmax(-1)) {
            return argmax.itemLong();
        }
    }

    @Override
    public String name() {
        return "greedy";
    }
}
