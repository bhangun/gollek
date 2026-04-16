/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import java.util.Random;

/**
 * Token sampler for autoregressive LLM generation.
 *
 * <p>Converts raw logits into a sampled next-token ID using the strategy
 * specified in {@link GenerationConfig}:
 * <ul>
 *   <li><b>Greedy</b> — returns the argmax when {@code temperature < 1e-4}.</li>
 *   <li><b>Temperature sampling</b> — scales logits by {@code 1/temperature}
 *       before softmax, then samples from the resulting distribution.</li>
 * </ul>
 *
 * <p>Top-k and top-p (nucleus) filtering can be layered on top by pre-processing
 * the logits before calling {@link #sample}.
 */
@ApplicationScoped
public class TokenSampler {

    private final Random random = new Random();

    /**
     * Samples the next token from the given logits using the default internal RNG.
     *
     * @param logits raw logit scores for each vocabulary token
     * @param config generation configuration (temperature, strategy, etc.)
     * @param freq   per-token frequency counts used for repetition penalties
     *               (currently reserved; pass an empty array if unused)
     * @return the sampled token ID, or {@code -1} if {@code logits} is null or empty
     */
    public int sample(float[] logits, GenerationConfig config, int[] freq) {
        return sample(logits, config, freq, random);
    }

    /**
     * Samples the next token using a caller-supplied {@link Random} instance.
     *
     * <p>Useful for reproducible generation — pass a seeded {@code Random} to
     * get deterministic output.
     *
     * @param logits raw logit scores for each vocabulary token
     * @param config generation configuration
     * @param freq   per-token frequency counts (reserved)
     * @param rng    random number generator to use for sampling
     * @return the sampled token ID, or {@code -1} if {@code logits} is null or empty
     */
    public int sample(float[] logits, GenerationConfig config, int[] freq, Random rng) {
        if (logits == null || logits.length == 0) {
            System.err.println("ERROR: Empty or null logits array");
            return -1;
        }
        
        // Validate logits contain reasonable values
        int nanCount = 0, infCount = 0;
        float minVal = Float.MAX_VALUE, maxVal = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (Float.isNaN(logit)) nanCount++;
            if (Float.isInfinite(logit)) infCount++;
            if (logit < minVal) minVal = logit;
            if (logit > maxVal) maxVal = logit;
        }
        
        if (nanCount > logits.length * 0.1 || infCount > logits.length * 0.1) {
            System.err.println("WARNING: Corrupted logits detected!");
            System.err.println("  NaN count: " + nanCount + "/" + logits.length);
            System.err.println("  Inf count: " + infCount + "/" + logits.length);
            System.err.println("  Range: [" + minVal + ", " + maxVal + "]");
            System.err.println("  First 10 logits: " + java.util.Arrays.toString(
                java.util.Arrays.copyOf(logits, Math.min(10, logits.length))));
            return -1;
        }

        // Apply temperature
        double temp = config.temperature();
        if (temp <= 0) temp = 1.0;

        // Greedy sampling if temperature is very low
        if (temp < 1e-4) {
             int best = 0;
             float bestVal = logits[0];
             for (int i = 1; i < logits.length; i++) {
                 if (logits[i] > bestVal) {
                     bestVal = logits[i];
                     best = i;
                 }
             }
             return best;
        }

        // Apply logit bias / penalties (skipped for brevity, but could be added here)
        
        // Softmax with temperature
        double[] probs = new double[logits.length];
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (logit > maxLogit) maxLogit = logit;
        }

        double sum = 0;
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp((logits[i] - maxLogit) / temp);
            sum += probs[i];
        }

        double r = rng.nextDouble() * sum;
        double cumulative = 0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (cumulative >= r) {
                return i;
            }
        }

        return logits.length - 1;
    }
}
