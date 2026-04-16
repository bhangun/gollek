/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioResampler.java
 * ───────────────────────
 * Audio resampling utilities.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import org.jboss.logging.Logger;

/**
 * High-quality audio resampling using sinc interpolation.
 * <p>
 * Supports arbitrary sample rate conversion with anti-aliasing filtering.
 * </p>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class AudioResampler {

    private static final Logger log = Logger.getLogger(AudioResampler.class);

    /**
     * Common sample rates.
     */
    public static final int[] COMMON_SAMPLE_RATES = {
            8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000
    };

    private final int sourceRate;
    private final int targetRate;
    private final int channels;
    private final float[] sincTable;

    /**
     * Create resampler.
     *
     * @param sourceRate source sample rate
     * @param targetRate target sample rate
     * @param channels   number of channels
     */
    public AudioResampler(int sourceRate, int targetRate, int channels) {
        this.sourceRate = sourceRate;
        this.targetRate = targetRate;
        this.channels = channels;
        this.sincTable = createSincTable(Math.max(sourceRate, targetRate));
    }

    /**
     * Create resampler for mono audio.
     *
     * @param sourceRate source sample rate
     * @param targetRate target sample rate
     */
    public AudioResampler(int sourceRate, int targetRate) {
        this(sourceRate, targetRate, 1);
    }

    /**
     * Resample audio to target sample rate.
     *
     * @param audio input audio samples
     * @return resampled audio
     */
    public float[] resample(float[] audio) {
        if (sourceRate == targetRate) {
            log.debug("No resampling needed (both " + sourceRate + " Hz)");
            return audio;
        }

        log.infof("Resampling: %d Hz → %d Hz, %d samples", sourceRate, targetRate, audio.length);

        double ratio = (double) targetRate / sourceRate;
        int outputLength = (int) (audio.length * ratio);
        float[] output = new float[outputLength];

        // Sinc interpolation
        int sincSize = sincTable.length / 2;

        for (int i = 0; i < outputLength; i++) {
            double srcPos = i / ratio;
            int srcIdx = (int) srcPos;
            double frac = srcPos - srcIdx;

            float sum = 0f;
            float weightSum = 0f;

            for (int j = -sincSize; j <= sincSize; j++) {
                int idx = srcIdx + j;
                if (idx >= 0 && idx < audio.length) {
                    double t = (j - frac) * ratio;
                    float weight = sincInterpolation(t);
                    sum += audio[idx] * weight;
                    weightSum += Math.abs(weight);
                }
            }

            output[i] = weightSum > 1e-6f ? sum / weightSum : 0f;
        }

        // Apply low-pass filter if downsampling
        if (targetRate < sourceRate) {
            applyLowPassFilter(output, targetRate);
        }

        log.debug("Resampled to " + outputLength + " samples (" + String.format("%.2f", ratio) + "x)");
        return output;
    }

    /**
     * Resample stereo audio.
     *
     * @param left  left channel
     * @param right right channel
     * @return resampled stereo audio [left, right]
     */
    public float[][] resampleStereo(float[] left, float[] right) {
        return new float[][] { resample(left), resample(right) };
    }

    /**
     * Convert sample format from int16 to float32.
     *
     * @param int16Data int16 audio data
     * @return float32 audio data normalized to [-1, 1]
     */
    public static float[] int16ToFloat32(short[] int16Data) {
        float[] float32Data = new float[int16Data.length];
        for (int i = 0; i < int16Data.length; i++) {
            float32Data[i] = int16Data[i] / 32768.0f;
        }
        return float32Data;
    }

    /**
     * Convert sample format from float32 to int16.
     *
     * @param float32Data float32 audio data normalized to [-1, 1]
     * @return int16 audio data
     */
    public static short[] float32ToInt16(float[] float32Data) {
        short[] int16Data = new short[float32Data.length];
        for (int i = 0; i < float32Data.length; i++) {
            float sample = Math.max(-1.0f, Math.min(1.0f, float32Data[i]));
            int16Data[i] = (short) (sample * 32767);
        }
        return int16Data;
    }

    /**
     * Normalize audio to [-1, 1] range.
     *
     * @param audio input audio
     * @return normalized audio
     */
    public static float[] normalize(float[] audio) {
        float maxAbs = 0f;
        for (float sample : audio) {
            float abs = Math.abs(sample);
            if (abs > maxAbs) maxAbs = abs;
        }

        if (maxAbs < 1e-6f) return audio;

        float[] normalized = new float[audio.length];
        for (int i = 0; i < audio.length; i++) {
            normalized[i] = audio[i] / maxAbs;
        }

        return normalized;
    }

    /**
     * Apply peak normalization to target level.
     *
     * @param audio       input audio
     * @param targetLevel target peak level (0.0 to 1.0)
     * @return normalized audio
     */
    public static float[] normalizeToLevel(float[] audio, float targetLevel) {
        float[] normalized = normalize(audio);

        if (targetLevel < 1.0f) {
            for (int i = 0; i < normalized.length; i++) {
                normalized[i] *= targetLevel;
            }
        }

        return normalized;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal methods
    // ─────────────────────────────────────────────────────────────────────────

    private float[] createSincTable(int maxRate) {
        int tableSize = 256; // Sinc table size
        float[] table = new float[tableSize * 2 + 1];

        // Windowed sinc function
        for (int i = -tableSize; i <= tableSize; i++) {
            double t = i / (double) tableSize;
            if (Math.abs(t) < 1e-6) {
                table[i + tableSize] = 1.0f;
            } else {
                // Sinc function with Hamming window
                double sinc = Math.sin(Math.PI * t) / (Math.PI * t);
                double window = 0.54 + 0.46 * Math.cos(2 * Math.PI * t);
                table[i + tableSize] = (float) (sinc * window);
            }
        }

        return table;
    }

    private float sincInterpolation(double t) {
        int tableSize = sincTable.length / 2;
        int idx = (int) (t * tableSize) + tableSize;

        if (idx < 0 || idx >= sincTable.length) {
            return 0f;
        }

        return sincTable[idx];
    }

    private void applyLowPassFilter(float[] audio, int sampleRate) {
        // Simple moving average low-pass filter
        int cutoff = Math.min(8000, sampleRate / 2 - 100);
        int windowSize = sampleRate / cutoff;

        if (windowSize < 2) return;

        float[] filtered = new float[audio.length];
        float sum = 0f;

        for (int i = 0; i < audio.length; i++) {
            sum += audio[i];

            if (i >= windowSize) {
                sum -= audio[i - windowSize];
            }

            filtered[i] = sum / Math.min(i + 1, windowSize);
        }

        System.arraycopy(filtered, 0, audio, 0, audio.length);
    }

    /**
     * Detect sample rate from audio data (heuristic).
     *
     * @param audio audio samples
     * @return estimated sample rate
     */
    public static int detectSampleRate(float[] audio) {
        // Simple heuristic based on audio duration assumptions
        // In production, would use more sophisticated detection
        int length = audio.length;

        // Assume audio is between 1 second and 10 minutes
        if (length < 8000) return 8000;
        if (length < 44100) return 11025;
        if (length < 96000) return 16000;
        if (length < 441000) return 22050;
        if (length < 960000) return 32000;
        return 44100;
    }

    /**
     * Get best common sample rate for target use case.
     *
     * @param useCase use case: "speech", "music", "general"
     * @return recommended sample rate
     */
    public static int getRecommendedSampleRate(String useCase) {
        return switch (useCase) {
            case "speech" -> 16000; // Whisper, SpeechT5
            case "music" -> 44100; // CD quality
            case "high_quality" -> 48000; // Professional
            default -> 22050; // General purpose
        };
    }
}
