package tech.kayys.gollek.inference.libtorch.sampling;

import tech.kayys.gollek.inference.libtorch.core.Tensor;

/**
 * Abstraction for token sampling strategies.
 * <p>
 * Given a logits tensor (shape [vocab_size] or [batch, vocab_size]),
 * produces the next token ID. Implementations may be deterministic
 * (greedy) or stochastic (temperature, top-k, top-p).
 */
public interface SamplingStrategy {

    /**
     * Sample the next token from the given logits.
     *
     * @param logits tensor of shape [vocab_size] or [1, vocab_size]
     * @return the selected token index
     */
    long sample(Tensor logits);

    /**
     * Human-readable name for this strategy (for logging/metrics).
     */
    String name();
}
