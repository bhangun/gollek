/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioProcessor.java
 * ───────────────────────
 * Audio processing utilities and effects.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import java.util.Arrays;

/**
 * Audio processing utilities for normalization, noise reduction, and effects.
 * 
 * <p>Provides common audio processing operations for improving audio quality
 * and preparing audio for transcription or analysis.</p>
 *
 * <h2>Features</h2>
 * <ul>
 *   <li>Peak and RMS normalization</li>
 *   <li>Noise gating</li>
 *   <li>Silence trimming</li>
 *   <li>DC offset removal</li>
 *   <li>Dynamic range compression</li>
 * </ul>
 *
 * @author Bhangun
 * @version 1.0.0
 * @since 2.1.0
 */
public final class AudioProcessor {

    private AudioProcessor() {
        // Utility class - prevent instantiation
    }

    /**
     * Normalize audio to peak level.
     *
     * @param audio input audio
     * @return normalized audio
     */
    public static float[] normalize(float[] audio) {
        return normalizeToLevel(audio, 1.0f);
    }

    /**
     * Normalize audio to specified peak level.
     *
     * @param audio input audio
     * @param targetLevel target peak level (0.0 to 1.0)
     * @return normalized audio
     */
    public static float[] normalizeToLevel(float[] audio, float targetLevel) {
        if (audio == null || audio.length == 0) {
            return audio;
        }

        // Find peak amplitude
        float peak = 0;
        for (float sample : audio) {
            float abs = Math.abs(sample);
            if (abs > peak) {
                peak = abs;
            }
        }

        // Already normalized or silent
        if (peak < 1e-6f || peak >= targetLevel) {
            return audio;
        }

        // Apply gain
        float gain = targetLevel / peak;
        float[] normalized = new float[audio.length];
        for (int i = 0; i < audio.length; i++) {
            normalized[i] = audio[i] * gain;
        }

        return normalized;
    }

    /**
     * Normalize audio to RMS level.
     *
     * @param audio input audio
     * @param targetRMS target RMS level (0.0 to 1.0)
     * @return normalized audio
     */
    public static float[] normalizeRMS(float[] audio, float targetRMS) {
        if (audio == null || audio.length == 0) {
            return audio;
        }

        // Calculate current RMS
        float sumSquares = 0;
        for (float sample : audio) {
            sumSquares += sample * sample;
        }
        float currentRMS = (float) Math.sqrt(sumSquares / audio.length);

        // Already at target or silent
        if (currentRMS < 1e-6f || currentRMS >= targetRMS) {
            return audio;
        }

        // Apply gain
        float gain = targetRMS / currentRMS;
        float[] normalized = new float[audio.length];
        for (int i = 0; i < audio.length; i++) {
            normalized[i] = audio[i] * gain;
        }

        return normalized;
    }

    /**
     * Apply noise gate to audio.
     *
     * @param audio input audio
     * @param threshold noise gate threshold (0.0 to 1.0)
     * @return gated audio
     */
    public static float[] noiseGate(float[] audio, float threshold) {
        if (audio == null || audio.length == 0) {
            return audio;
        }

        float[] gated = Arrays.copyOf(audio, audio.length);
        for (int i = 0; i < gated.length; i++) {
            if (Math.abs(gated[i]) < threshold) {
                gated[i] = 0;
            }
        }

        return gated;
    }

    /**
     * Remove DC offset from audio.
     *
     * @param audio input audio
     * @return audio with DC offset removed
     */
    public static float[] removeDCOffset(float[] audio) {
        if (audio == null || audio.length == 0) {
            return audio;
        }

        // Calculate mean (DC offset)
        float sum = 0;
        for (float sample : audio) {
            sum += sample;
        }
        float mean = sum / audio.length;

        // Remove DC offset
        float[] corrected = new float[audio.length];
        for (int i = 0; i < audio.length; i++) {
            corrected[i] = audio[i] - mean;
        }

        return corrected;
    }

    /**
     * Trim silence from beginning and end of audio.
     *
     * @param audio input audio
     * @param threshold silence threshold (0.0 to 1.0)
     * @return trimmed audio
     */
    public static float[] trimSilence(float[] audio, float threshold) {
        if (audio == null || audio.length == 0) {
            return audio;
        }

        // Find first non-silent sample
        int start = 0;
        while (start < audio.length && Math.abs(audio[start]) < threshold) {
            start++;
        }

        // Find last non-silent sample
        int end = audio.length - 1;
        while (end >= start && Math.abs(audio[end]) < threshold) {
            end--;
        }

        // All silence or no trimming needed
        if (start > end) {
            return new float[0];
        }
        if (start == 0 && end == audio.length - 1) {
            return audio;
        }

        return Arrays.copyOfRange(audio, start, end + 1);
    }

    /**
     * Apply dynamic range compression.
     *
     * @param audio input audio
     * @param threshold compression threshold in dB (e.g., -20)
     * @param ratio compression ratio (e.g., 4.0 for 4:1)
     * @param attack attack time in samples
     * @param release release time in samples
     * @return compressed audio
     */
    public static float[] compress(float[] audio, float threshold, float ratio,
                                   int attack, int release) {
        if (audio == null || audio.length == 0) {
            return audio;
        }

        float[] compressed = new float[audio.length];
        float gain = 1.0f;
        float thresholdLinear = (float) Math.pow(10, threshold / 20);

        for (int i = 0; i < audio.length; i++) {
            float inputLevel = Math.abs(audio[i]);
            float targetGain;

            if (inputLevel > thresholdLinear) {
                // Compress
                float excess = inputLevel / thresholdLinear;
                targetGain = (float) Math.pow(excess, (1 - ratio) / ratio) / ratio;
            } else {
                targetGain = 1.0f;
            }

            // Smooth gain changes
            if (targetGain < gain) {
                // Attack - reduce gain quickly
                gain = gain + (targetGain - gain) / attack;
            } else {
                // Release - increase gain slowly
                gain = gain + (targetGain - gain) / release;
            }

            compressed[i] = audio[i] * gain;
        }

        return compressed;
    }

    /**
     * Apply simple low-pass filter.
     *
     * @param audio input audio
     * @param cutoff normalized cutoff frequency (0.0 to 1.0, relative to Nyquist)
     * @return filtered audio
     */
    public static float[] lowPassFilter(float[] audio, float cutoff) {
        if (audio == null || audio.length == 0) {
            return audio;
        }

        // Simple first-order IIR low-pass filter
        float alpha = cutoff / (cutoff + 1);
        float[] filtered = new float[audio.length];
        filtered[0] = audio[0];

        for (int i = 1; i < audio.length; i++) {
            filtered[i] = filtered[i - 1] + alpha * (audio[i] - filtered[i - 1]);
        }

        return filtered;
    }

    /**
     * Apply simple high-pass filter.
     *
     * @param audio input audio
     * @param cutoff normalized cutoff frequency (0.0 to 1.0, relative to Nyquist)
     * @return filtered audio
     */
    public static float[] highPassFilter(float[] audio, float cutoff) {
        if (audio == null || audio.length == 0) {
            return audio;
        }

        // Simple first-order IIR high-pass filter
        float alpha = 1 / (cutoff + 1);
        float[] filtered = new float[audio.length];
        float lastInput = audio[0];
        filtered[0] = audio[0];

        for (int i = 1; i < audio.length; i++) {
            filtered[i] = alpha * (filtered[i - 1] + audio[i] - lastInput);
            lastInput = audio[i];
        }

        return filtered;
    }

    /**
     * Calculate RMS (Root Mean Square) level.
     *
     * @param audio input audio
     * @return RMS level
     */
    public static float calculateRMS(float[] audio) {
        if (audio == null || audio.length == 0) {
            return 0;
        }

        float sumSquares = 0;
        for (float sample : audio) {
            sumSquares += sample * sample;
        }

        return (float) Math.sqrt(sumSquares / audio.length);
    }

    /**
     * Calculate peak amplitude.
     *
     * @param audio input audio
     * @return peak amplitude
     */
    public static float calculatePeak(float[] audio) {
        if (audio == null || audio.length == 0) {
            return 0;
        }

        float peak = 0;
        for (float sample : audio) {
            float abs = Math.abs(sample);
            if (abs > peak) {
                peak = abs;
            }
        }

        return peak;
    }

    /**
     * Calculate zero-crossing rate.
     *
     * @param audio input audio
     * @return zero-crossing rate (crossings per sample)
     */
    public static float calculateZeroCrossingRate(float[] audio) {
        if (audio == null || audio.length < 2) {
            return 0;
        }

        int crossings = 0;
        for (int i = 1; i < audio.length; i++) {
            if ((audio[i - 1] >= 0 && audio[i] < 0) ||
                (audio[i - 1] < 0 && audio[i] >= 0)) {
                crossings++;
            }
        }

        return crossings / (float) audio.length;
    }

    /**
     * Detect clipping in audio.
     *
     * @param audio input audio
     * @param threshold clipping threshold (default 0.99)
     * @return true if clipping detected
     */
    public static boolean detectClipping(float[] audio, float threshold) {
        if (audio == null || audio.length == 0) {
            return false;
        }

        for (float sample : audio) {
            if (Math.abs(sample) > threshold) {
                return true;
            }
        }

        return false;
    }

    /**
     * Calculate clipping ratio.
     *
     * @param audio input audio
     * @param threshold clipping threshold
     * @return ratio of clipped samples (0.0 to 1.0)
     */
    public static float calculateClippingRatio(float[] audio, float threshold) {
        if (audio == null || audio.length == 0) {
            return 0;
        }

        int clipped = 0;
        for (float sample : audio) {
            if (Math.abs(sample) > threshold) {
                clipped++;
            }
        }

        return clipped / (float) audio.length;
    }
}
