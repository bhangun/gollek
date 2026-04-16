package tech.kayys.gollek.safetensor.engine.generation.attention;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Precomputed rotary positional embedding (RoPE) frequency cache.
 */
@ApplicationScoped
public class RopeFrequencyCache {

    private final Map<String, RopeFrequencies> cache = new ConcurrentHashMap<>();

    public RopeFrequencies get(ModelConfig config) {
        return get(config.resolvedHeadDim(), config.maxPositionEmbeddings(), config.ropeTheta());
    }

    public RopeFrequencies get(int rotaryDim, int maxSeqLen, double theta) {
        String key = rotaryDim + "-" + maxSeqLen + "-" + theta;
        return cache.computeIfAbsent(key, k -> new RopeFrequencies(rotaryDim, maxSeqLen, theta));
    }

    public static class RopeFrequencies {
        private final float[] cos;
        private final float[] sin;
        private final int rotaryDim;

        public RopeFrequencies(int rotaryDim, int maxSeqLen, double theta) {
            this.rotaryDim = rotaryDim;
            this.cos = new float[maxSeqLen * (rotaryDim / 2)];
            this.sin = new float[maxSeqLen * (rotaryDim / 2)];
            precompute(maxSeqLen, theta);
        }

        private void precompute(int maxSeqLen, double theta) {
            for (int i = 0; i < rotaryDim / 2; i++) {
                double freq = 1.0 / Math.pow(theta, (double) (2 * i) / rotaryDim);
                for (int t = 0; t < maxSeqLen; t++) {
                    double val = t * freq;
                    cos[t * (rotaryDim / 2) + i] = (float) Math.cos(val);
                    sin[t * (rotaryDim / 2) + i] = (float) Math.sin(val);
                }
            }
        }

        public void rotateInPlace(float[] x, int pos) {
            int half = rotaryDim / 2;
            int offset = pos * half;
            for (int i = 0; i < half; i++) {
                float x1 = x[i];
                float x2 = x[i + half];
                float c = cos[offset + i];
                float s = sin[offset + i];
                x[i] = x1 * c - x2 * s;
                x[i + half] = x1 * s + x2 * c;
            }
        }
    }
}
