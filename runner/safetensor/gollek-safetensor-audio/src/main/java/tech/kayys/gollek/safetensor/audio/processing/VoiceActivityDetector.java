/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * VoiceActivityDetector.java
 * ───────────────────────
 * Voice activity detection.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.List;

/**
 * Voice Activity Detection (VAD) for speech segmentation.
 * <p>
 * Detects speech regions in audio using energy-based and zero-crossing
 * rate analysis. Can be used to:
 * <ul>
 * <li>Remove silence from audio</li>
 * <li>Segment speech into utterances</li>
 * <li>Improve transcription efficiency</li>
 * </ul>
 * </p>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class VoiceActivityDetector {

    private static final Logger log = Logger.getLogger(VoiceActivityDetector.class);

    /**
     * Default energy threshold.
     */
    public static final float DEFAULT_ENERGY_THRESHOLD = 0.01f;

    /**
     * Default silence duration threshold (ms).
     */
    public static final int DEFAULT_SILENCE_THRESHOLD_MS = 500;

    /**
     * Default minimum speech duration (ms).
     */
    public static final int DEFAULT_MIN_SPEECH_MS = 250;

    private final float energyThreshold;
    private final int silenceThresholdMs;
    private final int minSpeechMs;
    private final int sampleRate;

    /**
     * Create VAD with default parameters.
     *
     * @param sampleRate audio sample rate
     */
    public VoiceActivityDetector(int sampleRate) {
        this(sampleRate, DEFAULT_ENERGY_THRESHOLD, DEFAULT_SILENCE_THRESHOLD_MS, DEFAULT_MIN_SPEECH_MS);
    }

    /**
     * Create VAD with custom parameters.
     *
     * @param sampleRate         audio sample rate
     * @param energyThreshold    energy threshold for voice detection
     * @param silenceThresholdMs silence duration to consider as non-speech
     * @param minSpeechMs        minimum speech duration to keep segment
     */
    public VoiceActivityDetector(int sampleRate, float energyThreshold, int silenceThresholdMs, int minSpeechMs) {
        this.sampleRate = sampleRate;
        this.energyThreshold = energyThreshold;
        this.silenceThresholdMs = silenceThresholdMs;
        this.minSpeechMs = minSpeechMs;
    }

    /**
     * Detect voice activity in audio.
     *
     * @param audio PCM audio samples
     * @return list of speech segments [start, end] in samples
     */
    public List<int[]> detectVoiceActivity(float[] audio) {
        log.debugf("Detecting voice activity in %d samples", audio.length);

        List<int[]> segments = new ArrayList<>();

        // Calculate frame energy
        int frameSize = sampleRate / 100; // 10ms frames
        int hopSize = frameSize / 2; // 5ms hop
        int numFrames = 1 + (audio.length - frameSize) / hopSize;

        float[] energy = new float[numFrames];
        boolean[] isVoice = new boolean[numFrames];

        // Calculate energy per frame
        for (int i = 0; i < numFrames; i++) {
            int start = i * hopSize;
            float sum = 0f;
            for (int j = 0; j < frameSize && start + j < audio.length; j++) {
                float sample = audio[start + j];
                sum += sample * sample;
            }
            energy[i] = (float) Math.sqrt(sum / frameSize);
        }

        // Adaptive threshold based on energy distribution
        float adaptiveThreshold = calculateAdaptiveThreshold(energy);

        // Classify frames as voice or non-voice
        for (int i = 0; i < numFrames; i++) {
            isVoice[i] = energy[i] > adaptiveThreshold;
        }

        // Apply median filter to smooth classification
        isVoice = medianFilter(isVoice, 3);

        // Find speech segments
        boolean inSpeech = false;
        int speechStart = 0;
        int silenceStart = 0;
        int silenceFrames = silenceThresholdMs / 5; // 5ms per frame

        for (int i = 0; i < numFrames; i++) {
            if (isVoice[i]) {
                if (!inSpeech) {
                    speechStart = i * hopSize;
                    inSpeech = true;
                }
                silenceStart = i;
            } else if (inSpeech) {
                // Check if silence duration exceeds threshold
                if (i - silenceStart > silenceFrames) {
                    int speechEnd = silenceStart * hopSize + frameSize;
                    int duration = speechEnd - speechStart;
                    int minSamples = minSpeechMs * sampleRate / 1000;

                    if (duration >= minSamples) {
                        segments.add(new int[] { speechStart, speechEnd });
                    }
                    inSpeech = false;
                }
            }
        }

        // Handle final segment
        if (inSpeech) {
            int speechEnd = Math.min(audio.length, numFrames * hopSize + frameSize);
            int duration = speechEnd - speechStart;
            int minSamples = minSpeechMs * sampleRate / 1000;

            if (duration >= minSamples) {
                segments.add(new int[] { speechStart, speechEnd });
            }
        }

        log.infof("Detected %d speech segments", segments.size());
        return segments;
    }

    /**
     * Remove silence from audio.
     *
     * @param audio input audio
     * @return audio with silence removed
     */
    public float[] removeSilence(float[] audio) {
        List<int[]> segments = detectVoiceActivity(audio);

        if (segments.isEmpty()) {
            log.warn("No speech detected - returning original audio");
            return audio;
        }

        // Calculate total speech duration
        int totalSpeechSamples = 0;
        for (int[] segment : segments) {
            totalSpeechSamples += segment[1] - segment[0];
        }

        // Concatenate speech segments
        float[] speech = new float[totalSpeechSamples];
        int pos = 0;

        for (int[] segment : segments) {
            int length = segment[1] - segment[0];
            System.arraycopy(audio, segment[0], speech, pos, length);
            pos += length;
        }

        log.infof("Removed silence: %d → %d samples", audio.length, speech.length);
        return speech;
    }

    /**
     * Split audio into utterances.
     *
     * @param audio input audio
     * @return list of utterance audio segments
     */
    public List<float[]> splitIntoUtterances(float[] audio) {
        List<int[]> segments = detectVoiceActivity(audio);
        List<float[]> utterances = new ArrayList<>();

        for (int[] segment : segments) {
            int length = segment[1] - segment[0];
            float[] utterance = new float[length];
            System.arraycopy(audio, segment[0], utterance, 0, length);
            utterances.add(utterance);
        }

        log.infof("Split into %d utterances", utterances.size());
        return utterances;
    }

    /**
     * Get speech ratio (percentage of audio that is speech).
     *
     * @param audio input audio
     * @return speech ratio (0.0 to 1.0)
     */
    public float getSpeechRatio(float[] audio) {
        List<int[]> segments = detectVoiceActivity(audio);

        int totalSpeechSamples = 0;
        for (int[] segment : segments) {
            totalSpeechSamples += segment[1] - segment[0];
        }

        return audio.length > 0 ? (float) totalSpeechSamples / audio.length : 0f;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal methods
    // ─────────────────────────────────────────────────────────────────────────

    private float calculateAdaptiveThreshold(float[] energy) {
        // Calculate percentile-based threshold
        float[] sorted = energy.clone();
        java.util.Arrays.sort(sorted);

        // Use 30th percentile as base threshold
        int idx = sorted.length * 30 / 100;
        float percentileThreshold = sorted[idx];

        // Also calculate mean energy
        float sum = 0f;
        for (float e : energy) sum += e;
        float meanEnergy = sum / energy.length;

        // Combine thresholds
        float adaptiveThreshold = Math.max(percentileThreshold, meanEnergy * 0.3f);

        // Ensure minimum threshold
        return Math.max(adaptiveThreshold, energyThreshold);
    }

    private boolean[] medianFilter(boolean[] data, int kernelSize) {
        boolean[] filtered = new boolean[data.length];
        int halfKernel = kernelSize / 2;

        for (int i = 0; i < data.length; i++) {
            int voiceCount = 0;
            int totalCount = 0;

            for (int j = -halfKernel; j <= halfKernel; j++) {
                int idx = i + j;
                if (idx >= 0 && idx < data.length) {
                    if (data[idx]) voiceCount++;
                    totalCount++;
                }
            }

            filtered[i] = voiceCount > totalCount / 2;
        }

        return filtered;
    }

    /**
     * Builder for VoiceActivityDetector.
     */
    public static class Builder {
        private int sampleRate = 16000;
        private float energyThreshold = DEFAULT_ENERGY_THRESHOLD;
        private int silenceThresholdMs = DEFAULT_SILENCE_THRESHOLD_MS;
        private int minSpeechMs = DEFAULT_MIN_SPEECH_MS;

        public Builder sampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        public Builder energyThreshold(float energyThreshold) {
            this.energyThreshold = energyThreshold;
            return this;
        }

        public Builder silenceThresholdMs(int silenceThresholdMs) {
            this.silenceThresholdMs = silenceThresholdMs;
            return this;
        }

        public Builder minSpeechMs(int minSpeechMs) {
            this.minSpeechMs = minSpeechMs;
            return this;
        }

        public VoiceActivityDetector build() {
            return new VoiceActivityDetector(sampleRate, energyThreshold, silenceThresholdMs, minSpeechMs);
        }
    }
}
