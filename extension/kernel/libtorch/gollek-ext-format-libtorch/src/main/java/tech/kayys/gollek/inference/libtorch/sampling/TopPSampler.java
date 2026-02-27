package tech.kayys.gollek.inference.libtorch.sampling;

import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Top-P (nucleus) sampling: dynamically selects the smallest set of tokens
 * whose cumulative probability exceeds a threshold P.
 * <p>
 * Unlike Top-K which uses a fixed count, Top-P adapts to the shape of the
 * distribution — selecting fewer tokens when one is dominant, and more
 * when the distribution is flat. This produces more natural text.
 */
public class TopPSampler implements SamplingStrategy {

    private final double p;
    private final double temperature;
    private final Random random;

    /**
     * @param p           cumulative probability threshold (0.0, 1.0]
     * @param temperature temperature for scaling (must be > 0)
     */
    public TopPSampler(double p, double temperature) {
        this(p, temperature, new Random());
    }

    public TopPSampler(double p, double temperature, Random random) {
        if (p <= 0 || p > 1.0)
            throw new IllegalArgumentException("p must be in (0, 1], got: " + p);
        if (temperature <= 0)
            throw new IllegalArgumentException("temperature must be > 0, got: " + temperature);
        this.p = p;
        this.temperature = temperature;
        this.random = random;
    }

    @Override
    public long sample(Tensor logits) {
        float[] logitValues = logits.toFloatArray();
        int vocabSize = logitValues.length;

        // Sort indices by descending logit
        Integer[] indices = new Integer[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(logitValues[b], logitValues[a]));

        // Compute softmax with temperature
        float maxLogit = logitValues[indices[0]];
        double[] probs = new double[vocabSize];
        double sumExp = 0.0;
        for (int i = 0; i < vocabSize; i++) {
            probs[i] = Math.exp((logitValues[indices[i]] - maxLogit) / temperature);
            sumExp += probs[i];
        }
        for (int i = 0; i < vocabSize; i++) {
            probs[i] /= sumExp;
        }

        // Find the nucleus (smallest set whose cumulative prob >= p)
        double cumulative = 0.0;
        int nucleusSize = 0;
        for (int i = 0; i < vocabSize; i++) {
            cumulative += probs[i];
            nucleusSize++;
            if (cumulative >= p) {
                break;
            }
        }

        // Re-normalize the nucleus probabilities
        double nucleusSum = 0.0;
        for (int i = 0; i < nucleusSize; i++) {
            nucleusSum += probs[i];
        }
        for (int i = 0; i < nucleusSize; i++) {
            probs[i] /= nucleusSum;
        }

        // Multinomial sample within nucleus
        double r = random.nextDouble();
        cumulative = 0.0;
        for (int i = 0; i < nucleusSize; i++) {
            cumulative += probs[i];
            if (r <= cumulative) {
                return indices[i];
            }
        }
        return indices[nucleusSize - 1]; // fallback
    }

    @Override
    public String name() {
        return "top_p(" + p + ", temp=" + temperature + ")";
    }
}
