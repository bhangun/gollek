/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.safetensor.engine.generation.attention;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import jakarta.enterprise.context.ApplicationScoped;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;

/**
 * Optimized cache for RoPE (Rotary Positional Embedding) frequencies.
 */
@ApplicationScoped
public class RopeFrequencyCache {

    private final Map<String, RopeFrequencies> cache = new ConcurrentHashMap<>();

    public RopeFrequencies get(int rotaryDim, int maxSeqLen, double theta, ModelConfig.RopeScaling scaling,
            int exponentDenominator, int rotatedDim) {
        String scalingKey = scaling == null ? "none" : scaling.type + "-" + scaling.factor;
        String key = rotaryDim + "-" + maxSeqLen + "-" + theta + "-" + scalingKey + "-" + exponentDenominator
                + "-" + rotatedDim;
        return cache.computeIfAbsent(key,
                k -> new RopeFrequencies(rotaryDim, maxSeqLen, theta, scaling, exponentDenominator, rotatedDim));
    }

    public static class RopeFrequencies {
        private final float[] cos;
        private final float[] sin;
        private final int rotaryDim;
        private final int rotatedDim;

        public RopeFrequencies(int rotaryDim, int maxSeqLen, double theta, ModelConfig.RopeScaling scaling,
                int exponentDenominator, int rotatedDim) {
            this.rotaryDim = rotaryDim;
            this.rotatedDim = rotatedDim;
            this.cos = new float[maxSeqLen * (rotaryDim / 2)];
            this.sin = new float[maxSeqLen * (rotaryDim / 2)];
            precompute(maxSeqLen, theta, scaling, exponentDenominator, rotatedDim);
        }

        private void precompute(int maxSeqLen, double theta, ModelConfig.RopeScaling scaling, int exponentDenominator,
                int rotatedDim) {
            int ropeDenominator = Math.max(2, exponentDenominator);
            int half = rotaryDim / 2;
            int rotatedHalf = Math.max(0, Math.min(half, rotatedDim / 2));
            for (int i = 0; i < half; i++) {
                if (i >= rotatedHalf) {
                    for (int t = 0; t < maxSeqLen; t++) {
                        cos[t * half + i] = 1.0f;
                        sin[t * half + i] = 0.0f;
                    }
                    continue;
                }
                double freq = 1.0 / Math.pow(theta, (double) (2 * i) / ropeDenominator);

                if (scaling != null && "llama3".equalsIgnoreCase(scaling.type)) {
                    double wavelen = 2 * Math.PI / freq;
                    double lowFreqWavelen = scaling.originalMaxPositionEmbeddings / scaling.lowFreqFactor;
                    double highFreqWavelen = scaling.originalMaxPositionEmbeddings / scaling.highFreqFactor;
                    
                    if (wavelen > lowFreqWavelen) {
                        freq = freq / scaling.factor;
                    } else if (wavelen > highFreqWavelen) {
                        double smooth = (scaling.originalMaxPositionEmbeddings / wavelen - scaling.lowFreqFactor) /
                                (scaling.highFreqFactor - scaling.lowFreqFactor);
                        freq = (1 - smooth) * freq / scaling.factor + smooth * freq;
                    }
                } else if (scaling != null && "linear".equalsIgnoreCase(scaling.type)) {
                    freq = freq / scaling.factor;
                }

                for (int t = 0; t < maxSeqLen; t++) {
                    double val = t * freq;
                    cos[t * half + i] = (float) Math.cos(val);
                    sin[t * half + i] = (float) Math.sin(val);
                }
            }
        }

        /**
         * Rotates a segment slice in place.
         * @param interleaved if true, uses [x1, x2, x3, x4] -> [x1*c-x2*s, x1*s+x2*c, ...] (Gemma/Llama)
         *                    if false, uses [x1...xn/2, xn/2+1...xn] -> [x1*c-xn/2+1*s, ...] (GPT-NeoX)
         */
        public void rotateInPlace(MemorySegment seg, long elementOffset, int pos, boolean interleaved) {
            int half = rotaryDim / 2;
            int freqOffset = pos * half;
            
            if (interleaved) {
                // Interleaved RoPE (Llama/Gemma Style)
                for (int i = 0; i < half; i++) {
                    float xv1 = seg.getAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + i * 2);
                    float xv2 = seg.getAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + i * 2 + 1);
                    float cv = cos[freqOffset + i];
                    float sv = sin[freqOffset + i];
                    seg.setAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + i * 2, xv1 * cv - xv2 * sv);
                    seg.setAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + i * 2 + 1, xv1 * sv + xv2 * cv);
                }
            } else {
                // Split-Half RoPE (GPT-NeoX Style)
                int rotatedHalf = Math.max(0, Math.min(half, rotatedDim / 2));
                for (int i = 0; i < half; i++) {
                    if (i >= rotatedHalf) {
                        // These elements are passed through unmodified in partial RoPE.
                        // For HuggingFace, they are at the END of the vector (after rotatedDim).
                        // Wait, if rotatedDim < rotaryDim, the unrotated elements start at rotatedDim!
                        // So for i >= rotatedHalf, the index should be `rotatedDim + (i - rotatedHalf) * 2`? No!
                        // In HF: q_rot is `0..rotatedDim-1`. q_pass is `rotatedDim..headDim-1`.
                        // But wait! If we just don't modify them, they stay where they are.
                        // Are they already in the correct place? Yes! 
                        continue;
                    }
                    float xv1 = seg.getAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + i);
                    float xv2 = seg.getAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + i + rotatedHalf);
                    float cv = cos[freqOffset + i];
                    float sv = sin[freqOffset + i];
                    seg.setAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + i, xv1 * cv - xv2 * sv);
                    seg.setAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + i + rotatedHalf, xv1 * sv + xv2 * cv);
                }
            }
        }

        public void rotateInPlace(float[] x, int pos, boolean interleaved, int rotatedDim) {
            int half = rotaryDim / 2;
            int offset = pos * half;
            if (interleaved) {
                int rotatedHalf = Math.max(0, Math.min(half, rotatedDim / 2));
                for (int i = 0; i < half; i++) {
                    if (i >= rotatedHalf) {
                        continue;
                    }
                    float x1 = x[i * 2];
                    float x2 = x[i * 2 + 1];
                    float c = cos[offset + i];
                    float s = sin[offset + i];
                    x[i * 2] = x1 * c - x2 * s;
                    x[i * 2 + 1] = x1 * s + x2 * c;
                }
            } else {
                int rotatedHalf = Math.max(0, Math.min(half, rotatedDim / 2));
                for (int i = 0; i < half; i++) {
                    if (i >= rotatedHalf) {
                        continue;
                    }
                    float x1 = x[i];
                    float x2 = x[i + rotatedHalf];
                    float c = cos[offset + i];
                    float s = sin[offset + i];
                    x[i] = x1 * c - x2 * s;
                    x[i + rotatedHalf] = x1 * s + x2 * c;
                }
            }
        }
    }
}
