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

        // Extract values
        double temp = config.temperature();
        int topK = config.topK();
        float topP = config.topP();
        float minP = config.minP();

        // Apply repetition and frequency penalties (applies to both greedy and sampled modes)
        float repPenalty = config.repetitionPenalty();
        float freqPenalty = config.frequencyPenalty();
        if (freq != null && (repPenalty > 1.0f || freqPenalty > 0.0f)) {
            for (int i = 0; i < logits.length && i < freq.length; i++) {
                if (freq[i] > 0) {
                    if (repPenalty > 1.0f) {
                        if (logits[i] > 0) {
                            logits[i] /= repPenalty;
                        } else {
                            logits[i] *= repPenalty;
                        }
                    }
                    if (freqPenalty > 0.0f) {
                        logits[i] -= freqPenalty * freq[i];
                    }
                }
            }
        }

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

        if (temp <= 0) temp = 1.0;

        // Create an index array
        int[] indices = new int[logits.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }

        // Sort indices by logit (Descending) - we only need to sort roughly topK + 1,
        // but sorting the array is generally fast enough.
        quickSortIndices(logits, indices, 0, logits.length - 1);

        // Max logit is now at indices[0]
        float maxLogit = logits[indices[0]];

        // Softmax with temperature
        double[] probs = new double[logits.length];
        double sum = 0;
        
        // Only evaluate Top-K if topK is valid and < vocab_size
        int limit = (topK > 0 && topK < logits.length) ? topK : logits.length;

        // Apply Min-P limit first
        double minProbLimit = 0.0;
        if (minP > 0.0f) {
             minProbLimit = minP * Math.exp((maxLogit - maxLogit) / temp); // which is just minP
        }

        int actualElements = 0;
        for (int i = 0; i < limit; i++) {
            int idx = indices[i];
            double p = Math.exp((logits[idx] - maxLogit) / temp);
            
            if (minP > 0.0f && p < minProbLimit && actualElements > 0) {
                // If it falls below threshold and we already have at least 1 token, stop.
                break;
            }
            
            probs[i] = p;
            sum += p;
            actualElements++;
        }

        // Top-P filter
        if (topP > 0.0f && topP < 1.0f) {
            double cumulative = 0.0;
            for (int i = 0; i < actualElements; i++) {
                double pNorm = probs[i] / sum;
                cumulative += pNorm;
                if (cumulative > topP && i > 0) {
                    actualElements = i + 1;
                    break;
                }
            }
        }

        // Recompute sum over filtered elements for exact normalization
        double filteredSum = 0;
        for (int i = 0; i < actualElements; i++) {
            filteredSum += probs[i];
        }

        double r = rng.nextDouble() * filteredSum;
        double cumulative = 0;
        for (int i = 0; i < actualElements; i++) {
            cumulative += probs[i];
            if (cumulative >= r) {
                return indices[i];
            }
        }

        return indices[0];
    }

    private void quickSortIndices(float[] values, int[] indices, int low, int high) {
        if (low < high) {
            int pi = partition(values, indices, low, high);
            quickSortIndices(values, indices, low, pi - 1);
            quickSortIndices(values, indices, pi + 1, high);
        }
    }

    private int partition(float[] values, int[] indices, int low, int high) {
        float pivot = values[indices[high]];
        int i = (low - 1);
        for (int j = low; j < high; j++) {
            if (values[indices[j]] > pivot) { // Descending order
                i++;
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        int temp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp;
        return i + 1;
    }
}
