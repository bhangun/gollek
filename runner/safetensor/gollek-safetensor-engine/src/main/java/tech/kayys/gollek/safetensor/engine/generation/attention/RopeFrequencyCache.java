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
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import jakarta.enterprise.context.ApplicationScoped;

import tech.kayys.gollek.spi.model.ModelConfig;

/**
 * Optimized cache for RoPE (Rotary Positional Embedding) frequencies.
 */
@ApplicationScoped
public class RopeFrequencyCache {

    private final Map<String, RopeFrequencies> cache = new ConcurrentHashMap<>();

    public RopeFrequencies get(int rotaryDim, int maxSeqLen, double theta, ModelConfig.RopeScaling scaling) {
        String scalingKey = scaling == null ? "none" : scaling.type + "-" + scaling.factor;
        String key = rotaryDim + "-" + maxSeqLen + "-" + theta + "-" + scalingKey;
        return cache.computeIfAbsent(key, k -> new RopeFrequencies(rotaryDim, maxSeqLen, theta, scaling));
    }

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static class RopeFrequencies {
        private final float[] cos;
        private final float[] sin;
        private final int rotaryDim;

        public RopeFrequencies(int rotaryDim, int maxSeqLen, double theta, ModelConfig.RopeScaling scaling) {
            this.rotaryDim = rotaryDim;
            this.cos = new float[maxSeqLen * (rotaryDim / 2)];
            this.sin = new float[maxSeqLen * (rotaryDim / 2)];
            precompute(maxSeqLen, theta, scaling);
        }

        private void precompute(int maxSeqLen, double theta, ModelConfig.RopeScaling scaling) {
            for (int i = 0; i < rotaryDim / 2; i++) {
                double freq = 1.0 / Math.pow(theta, (double) (2 * i) / rotaryDim);

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
                    cos[t * (rotaryDim / 2) + i] = (float) Math.cos(val);
                    sin[t * (rotaryDim / 2) + i] = (float) Math.sin(val);
                }
            }
        }

        /**
         * Rotates a segment slice in place.
         * Qwen 2.5 uses INTERLEAVED RoPE: (x0, x1) paired.
         */
        public void rotateInPlace(MemorySegment seg, long elementOffset, int pos) {
            int half = rotaryDim / 2;
            int freqOffset = pos * half;
            
            // Vectorized Split-Half RoPE (Llama/Qwen Style)
            // x1 = x[0...half], x2 = x[half...d]
            for (int i = 0; i < half; i += SPECIES.length()) {
                int limit = Math.min(i + SPECIES.length(), half);
                int num = limit - i;
                
                if (num == SPECIES.length()) {
                    FloatVector x1 = FloatVector.fromMemorySegment(SPECIES, seg, (elementOffset + i) * 4, ByteOrder.LITTLE_ENDIAN);
                    FloatVector x2 = FloatVector.fromMemorySegment(SPECIES, seg, (elementOffset + i + half) * 4, ByteOrder.LITTLE_ENDIAN);
                    FloatVector c = FloatVector.fromArray(SPECIES, cos, freqOffset + i);
                    FloatVector s = FloatVector.fromArray(SPECIES, sin, freqOffset + i);
                    
                    // res1 = x1*c - x2*s
                    // res2 = x1*s + x2*c
                    x1.mul(c).sub(x2.mul(s)).intoMemorySegment(seg, (elementOffset + i) * 4, ByteOrder.LITTLE_ENDIAN);
                    x1.mul(s).add(x2.mul(c)).intoMemorySegment(seg, (elementOffset + i + half) * 4, ByteOrder.LITTLE_ENDIAN);
                } else {
                    for (int j = i; j < limit; j++) {
                        float xv1 = seg.getAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + j);
                        float xv2 = seg.getAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + j + half);
                        float cv = cos[freqOffset + j];
                        float sv = sin[freqOffset + j];
                        seg.setAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + j, xv1 * cv - xv2 * sv);
                        seg.setAtIndex(ValueLayout.JAVA_FLOAT, elementOffset + j + half, xv1 * sv + xv2 * cv);
                    }
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
