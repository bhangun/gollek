package tech.kayys.gollek.inference.libtorch.sampling;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Top-K sampling: restricts sampling to the K highest-probability tokens.
 * <p>
 * After selecting the top K tokens, applies temperature scaling and
 * samples from the truncated distribution. This prevents sampling from
 * the long tail of unlikely tokens while preserving diversity.
 */
public class TopKSampler implements SamplingStrategy {

    private final int k;
    private final double temperature;
    private final Random random;

    /**
     * @param k           number of top tokens to consider (must be >= 1)
     * @param temperature temperature for scaling (must be > 0)
     */
    public TopKSampler(int k, double temperature) {
        this(k, temperature, new Random());
    }

    public TopKSampler(int k, double temperature, Random random) {
        if (k < 1)
            throw new IllegalArgumentException("k must be >= 1, got: " + k);
        if (temperature <= 0)
            throw new IllegalArgumentException("temperature must be > 0, got: " + temperature);
        this.k = k;
        this.temperature = temperature;
        this.random = random;
    }

    @Override
    public long sample(TorchTensor logits) {
        float[] logitValues = logits.toFloatArray();
        int vocabSize = logitValues.length;
        int effectiveK = Math.min(k, vocabSize);

        // Find the top-K indices by sorting index array
        Integer[] indices = new Integer[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(logitValues[b], logitValues[a]));

        // Build truncated distribution over top-K
        double[] probs = new double[effectiveK];
        float maxLogit = logitValues[indices[0]];

        double sumExp = 0.0;
        for (int i = 0; i < effectiveK; i++) {
            probs[i] = Math.exp((logitValues[indices[i]] - maxLogit) / temperature);
            sumExp += probs[i];
        }
        for (int i = 0; i < effectiveK; i++) {
            probs[i] /= sumExp;
        }

        // Multinomial sample within top-K
        double r = random.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < effectiveK; i++) {
            cumulative += probs[i];
            if (r <= cumulative) {
                return indices[i];
            }
        }
        return indices[effectiveK - 1]; // fallback
    }

    @Override
    public String name() {
        return "top_k(" + k + ", temp=" + temperature + ")";
    }
}
