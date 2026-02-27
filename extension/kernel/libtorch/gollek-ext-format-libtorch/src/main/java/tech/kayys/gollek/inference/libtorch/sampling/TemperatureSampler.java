package tech.kayys.gollek.inference.libtorch.sampling;

import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.util.Random;

/**
 * Temperature-scaled sampling.
 * <p>
 * Divides logits by a temperature value before applying softmax and sampling.
 * Higher temperature → more random output; lower temperature → more focused.
 * Temperature of 1.0 is neutral; temperature of 0 degenerates to greedy.
 */
public class TemperatureSampler implements SamplingStrategy {

    private final double temperature;
    private final Random random;

    /**
     * @param temperature scaling factor (must be > 0). Values < 1.0
     *                    produce sharper distributions, > 1.0 flatter.
     */
    public TemperatureSampler(double temperature) {
        this(temperature, new Random());
    }

    public TemperatureSampler(double temperature, Random random) {
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be > 0, got: " + temperature);
        }
        this.temperature = temperature;
        this.random = random;
    }

    @Override
    public long sample(Tensor logits) {
        // If temperature is very close to 0, fall back to greedy
        if (temperature < 1e-7) {
            try (Tensor argmax = logits.argmax(-1)) {
                return argmax.itemLong();
            }
        }

        float[] logitValues = logits.toFloatArray();

        // Apply temperature scaling
        for (int i = 0; i < logitValues.length; i++) {
            logitValues[i] /= temperature;
        }

        // Softmax in Java (stable version)
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float v : logitValues) {
            if (v > maxLogit)
                maxLogit = v;
        }

        double sumExp = 0.0;
        double[] probs = new double[logitValues.length];
        for (int i = 0; i < logitValues.length; i++) {
            probs[i] = Math.exp(logitValues[i] - maxLogit);
            sumExp += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sumExp;
        }

        // Multinomial sample
        return multinomialSample(probs);
    }

    private long multinomialSample(double[] probs) {
        double r = random.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) {
                return i;
            }
        }
        return probs.length - 1; // fallback
    }

    @Override
    public String name() {
        return "temperature(" + temperature + ")";
    }
}
