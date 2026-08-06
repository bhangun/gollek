/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Random;

/**
 * Token sampler for autoregressive LLM generation.
 *
 * <p>
 * Converts raw logits into a sampled next-token ID using the strategy
 * specified in {@link GenerationConfig}:
 * <ul>
 * <li><b>Greedy</b> — returns the argmax when requested by strategy,
 * {@code temperature < 1e-4}, or {@code topK == 1}.</li>
 * <li><b>Temperature sampling</b> — scales logits by {@code 1/temperature}
 * before softmax, then samples from the resulting distribution.</li>
 * </ul>
 *
 * <p>
 * Top-k, top-p (nucleus), and min-p filtering are delegated to the sampling
 * distribution helper so this class remains focused on token-selection flow.
 */
@ApplicationScoped
public class TokenSampler {

    private final Random random = new Random();
    private final TokenSamplingDistribution distribution = new TokenSamplingDistribution();

    /**
     * Samples the next token from the given logits using the default internal RNG.
     *
     * @param logits raw logit scores for each vocabulary token
     * @param config generation configuration (temperature, strategy, etc.)
     * @param freq   per-token frequency counts used for repetition penalties
     *               (currently reserved; pass an empty array if unused)
     * @return the sampled token ID, or {@code -1} if {@code logits} is null or
     *         empty
     */
    public int sample(float[] logits, GenerationConfig config, int[] freq) {
        return sample(logits, config, null, freq);
    }

    public int sample(float[] logits, GenerationConfig config, ModelConfig modelConfig, int[] freq) {
        return sample(logits, config, modelConfig, freq, random);
    }

    /**
     * Samples the next token using a caller-supplied {@link Random} instance.
     *
     * <p>
     * Useful for reproducible generation — pass a seeded {@code Random} to
     * get deterministic output.
     *
     * @param logits raw logit scores for each vocabulary token
     * @param config generation configuration
     * @param freq   per-token frequency counts (reserved)
     * @param rng    random number generator to use for sampling
     * @return the sampled token ID, or {@code -1} if {@code logits} is null or
     *         empty
     */
    public int sample(float[] logits, GenerationConfig config, ModelConfig modelConfig, int[] freq, Random rng) {
        if (!TokenSamplingLogitValidator.accepts(logits)) {
            return -1;
        }

        TokenSamplingPenaltyPolicy.apply(logits, freq, config);

        // Greedy sampling when the effective sampling policy is deterministic.
        if (config.requestsGreedyDecoding()) {
            return GenerationGreedyArgmax.selectJava(logits);
        }

        return distribution.sample(logits, config, modelConfig, rng);
    }
}
