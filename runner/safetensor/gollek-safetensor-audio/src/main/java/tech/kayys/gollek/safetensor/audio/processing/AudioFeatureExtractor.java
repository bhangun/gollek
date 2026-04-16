/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioFeatureExtractor.java
 * ───────────────────────
 * Audio feature extraction utilities.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import org.jboss.logging.Logger;

/**
 * Extract audio features for speech processing models.
 * <p>
 * Supports:
 * <ul>
 * <li>Log-Mel spectrogram extraction</li>
 * <li>MFCC (Mel-frequency cepstral coefficients)</li>
 * <li>F0 (fundamental frequency) estimation</li>
 * <li>Energy extraction</li>
 * </ul>
 * </p>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class AudioFeatureExtractor {

    private static final Logger log = Logger.getLogger(AudioFeatureExtractor.class);

    /**
     * Default sample rate (16kHz for speech).
     */
    public static final int DEFAULT_SAMPLE_RATE = 16000;

    /**
     * Default FFT size.
     */
    public static final int DEFAULT_N_FFT = 400;

    /**
     * Default hop length.
     */
    public static final int DEFAULT_HOP_LENGTH = 160;

    /**
     * Default number of mel bins.
     */
    public static final int DEFAULT_N_MEL = 80;

    private final int sampleRate;
    private final int nFft;
    private final int hopLength;
    private final int nMel;
    private final float[][] melFilterbank;

    /**
     * Create feature extractor with default parameters.
     */
    public AudioFeatureExtractor() {
        this(DEFAULT_SAMPLE_RATE, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_N_MEL);
    }

    /**
     * Create feature extractor with custom parameters.
     *
     * @param sampleRate sample rate in Hz
     * @param nFft       FFT size
     * @param hopLength  hop length in samples
     * @param nMel       number of mel bins
     */
    public AudioFeatureExtractor(int sampleRate, int nFft, int hopLength, int nMel) {
        this.sampleRate = sampleRate;
        this.nFft = nFft;
        this.hopLength = hopLength;
        this.nMel = nMel;
        this.melFilterbank = createMelFilterbank(nMel, nFft, sampleRate);
    }

    /**
     * Extract log-mel spectrogram from PCM audio.
     *
     * @param pcm        PCM audio samples (float32, normalized to [-1, 1])
     * @param nMels      number of mel bins
     * @param nFft       FFT size
     * @param hopLength  hop length
     * @param maxFrames  maximum number of frames (0 = unlimited)
     * @return log-mel spectrogram [nMels, nFrames]
     */
    public float[][] extractLogMelSpectrogram(float[] pcm, int nMels, int nFft, int hopLength, int maxFrames) {
        log.debugf("Extracting log-mel spectrogram: %d samples, %d mel bins", pcm.length, nMels);

        // Pad or truncate to appropriate length
        int targetSamples = maxFrames > 0 ? maxFrames * hopLength + nFft : pcm.length + nFft;
        float[] padded = padOrTrim(pcm, targetSamples);

        // Apply window function
        float[] window = hannWindow(nFft);

        // Calculate number of frames
        int numFrames = 1 + (padded.length - nFft) / hopLength;
        float[][] spectrogram = new float[nMels][numFrames];

        // Process each frame
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopLength;

            // Apply window and extract frame
            float[] framed = new float[nFft];
            for (int i = 0; i < nFft && start + i < padded.length; i++) {
                framed[i] = padded[start + i] * window[i];
            }

            // Compute FFT magnitude
            float[] magnitude = fftMagnitude(framed);

            // Apply mel filterbank
            for (int m = 0; m < nMels; m++) {
                float energy = 0f;
                for (int k = 0; k < magnitude.length; k++) {
                    energy += magnitude[k] * melFilterbank[m][k];
                }
                // Log compression with epsilon for numerical stability
                spectrogram[m][frame] = (float) Math.log10(Math.max(1e-10f, energy));
            }
        }

        // Normalize to [-1, 1] range (Whisper-style)
        normalizeLogMel(spectrogram);

        return spectrogram;
    }

    /**
     * Extract log-mel spectrogram with default parameters.
     *
     * @param pcm PCM audio samples
     * @return log-mel spectrogram
     */
    public float[][] extractLogMelSpectrogram(float[] pcm) {
        return extractLogMelSpectrogram(pcm, nMel, nFft, hopLength, 0);
    }

    /**
     * Extract MFCC features.
     *
     * @param pcm        PCM audio samples
     * @param numCepstra number of cepstral coefficients
     * @return MFCC features [numCepstra, nFrames]
     */
    public float[][] extractMFCC(float[] pcm, int numCepstra) {
        log.debugf("Extracting MFCC: %d samples, %d cepstra", pcm.length, numCepstra);

        // Get log-mel spectrogram
        float[][] logMel = extractLogMelSpectrogram(pcm);

        int numFrames = logMel[0].length;
        float[][] mfcc = new float[numCepstra][numFrames];

        // Apply DCT-II to each frame
        for (int f = 0; f < numFrames; f++) {
            for (int c = 0; c < numCepstra; c++) {
                float sum = 0f;
                for (int m = 0; m < nMel; m++) {
                    sum += logMel[m][f] * (float) Math.cos(Math.PI * c * (m + 0.5) / nMel);
                }
                mfcc[c][f] = (float) (sum * Math.sqrt(2.0 / nMel));
            }
        }

        return mfcc;
    }

    /**
     * Extract F0 (fundamental frequency) using autocorrelation.
     *
     * @param pcm       PCM audio samples
     * @param sampleRate sample rate
     * @param minF0     minimum F0 in Hz
     * @param maxF0     maximum F0 in Hz
     * @return F0 contour in Hz
     */
    public float[] extractF0(float[] pcm, int sampleRate, float minF0, float maxF0) {
        log.debugf("Extracting F0: range [%.1f, %.1f] Hz", minF0, maxF0);

        int frameSize = sampleRate / 100; // 10ms frames
        int hopSize = frameSize / 2; // 5ms hop
        int numFrames = 1 + (pcm.length - frameSize) / hopSize;
        float[] f0 = new float[numFrames];

        int minLag = sampleRate / (int) maxF0;
        int maxLag = sampleRate / (int) minF0;

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            float[] frameData = new float[frameSize];
            System.arraycopy(pcm, start, frameData, 0, Math.min(frameSize, pcm.length - start));

            // Compute autocorrelation
            float[] autocorr = autocorrelation(frameData);

            // Find peak in valid lag range
            float maxCorr = 0f;
            int bestLag = 0;
            for (int lag = minLag; lag < Math.min(maxLag, autocorr.length); lag++) {
                if (autocorr[lag] > maxCorr) {
                    maxCorr = autocorr[lag];
                    bestLag = lag;
                }
            }

            // Convert lag to F0
            f0[frame] = bestLag > 0 ? sampleRate / (float) bestLag : 0f;
        }

        return f0;
    }

    /**
     * Extract energy contour.
     *
     * @param pcm PCM audio samples
     * @return energy contour (RMS)
     */
    public float[] extractEnergy(float[] pcm) {
        int frameSize = sampleRate / 100; // 10ms frames
        int hopSize = frameSize / 2;
        int numFrames = 1 + (pcm.length - frameSize) / hopSize;
        float[] energy = new float[numFrames];

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopSize;
            float sum = 0f;
            for (int i = 0; i < frameSize && start + i < pcm.length; i++) {
                float sample = pcm[start + i];
                sum += sample * sample;
            }
            energy[frame] = (float) Math.sqrt(sum / frameSize);
        }

        return energy;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal helper methods
    // ─────────────────────────────────────────────────────────────────────────

    private float[][] createMelFilterbank(int nMel, int nFft, int sampleRate) {
        float[] melFreqs = melFilterFreqs(nMel, sampleRate);
        float[][] filterbank = new float[nMel][nFft / 2 + 1];

        for (int m = 0; m < nMel; m++) {
            for (int k = 0; k <= nFft / 2; k++) {
                filterbank[m][k] = melWeight(k, m, melFreqs, sampleRate, nFft);
            }
        }

        return filterbank;
    }

    private float[] melFilterFreqs(int nMel, int sampleRate) {
        float melMin = hzToMel(0f);
        float melMax = hzToMel(sampleRate / 2.0f);
        float[] freqs = new float[nMel + 2];

        for (int i = 0; i < freqs.length; i++) {
            freqs[i] = melToHz(melMin + i * (melMax - melMin) / (nMel + 1));
        }

        return freqs;
    }

    private float melWeight(int bin, int mel, float[] freqs, int sr, int nFft) {
        float hz = bin * sr / (float) nFft;
        float lower = freqs[mel];
        float center = freqs[mel + 1];
        float upper = freqs[mel + 2];

        if (hz < lower || hz >= upper)
            return 0f;
        if (hz < center)
            return (hz - lower) / (center - lower);
        return (upper - hz) / (upper - center);
    }

    private static float hzToMel(float hz) {
        return 2595f * (float) Math.log10(1f + hz / 700f);
    }

    private static float melToHz(float mel) {
        return 700f * ((float) Math.pow(10, mel / 2595f) - 1f);
    }

    private float[] hannWindow(int size) {
        float[] window = new float[size];
        for (int i = 0; i < size; i++) {
            window[i] = 0.5f * (1f - (float) Math.cos(2 * Math.PI * i / (size - 1)));
        }
        return window;
    }

    private float[] padOrTrim(float[] pcm, int targetLength) {
        if (pcm.length == targetLength)
            return pcm;

        float[] result = new float[targetLength];
        System.arraycopy(pcm, 0, result, 0, Math.min(pcm.length, targetLength));
        return result;
    }

    private void normalizeLogMel(float[][] spectrogram) {
        float maxVal = Float.NEGATIVE_INFINITY;
        float minVal = Float.MAX_VALUE;

        for (float[] row : spectrogram) {
            for (float v : row) {
                if (v > maxVal) maxVal = v;
                if (v < minVal) minVal = v;
            }
        }

        float range = maxVal - minVal;
        if (range < 1e-6f) range = 1f;

        // Scale to [-1, 1]
        for (int m = 0; m < spectrogram.length; m++) {
            for (int f = 0; f < spectrogram[m].length; f++) {
                float v = spectrogram[m][f];
                v = (v - minVal) / range; // [0, 1]
                v = v * 2 - 1; // [-1, 1]
                spectrogram[m][f] = Math.max(-1f, Math.min(1f, v));
            }
        }
    }

    private float[] fftMagnitude(float[] x) {
        int n = x.length;
        if ((n & (n - 1)) != 0) {
            // Pad to next power of 2
            int nextPow2 = Integer.highestOneBit(n) << 1;
            x = padOrTrim(x, nextPow2);
            n = nextPow2;
        }

        float[] real = java.util.Arrays.copyOf(x, n);
        float[] imag = new float[n];

        // Cooley-Tukey FFT
        fft(real, imag);

        // Compute magnitude for first N/2+1 bins
        float[] mag = new float[n / 2 + 1];
        for (int i = 0; i <= n / 2; i++) {
            mag[i] = (float) Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
        }

        return mag;
    }

    private void fft(float[] real, float[] imag) {
        int n = real.length;

        // Bit-reversal permutation
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j) {
                float t = real[i];
                real[i] = real[j];
                real[j] = t;
                t = imag[i];
                imag[i] = imag[j];
                imag[j] = t;
            }
        }

        // FFT butterfly
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2 * Math.PI / len;
            float wr = (float) Math.cos(ang), wi = (float) Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                float cr = 1f, ci = 0f;
                for (int j = 0; j < len / 2; j++) {
                    float ur = real[i + j], ui = imag[i + j];
                    float vr = real[i + j + len / 2] * cr - imag[i + j + len / 2] * ci;
                    float vi = real[i + j + len / 2] * ci + imag[i + j + len / 2] * cr;
                    real[i + j] = ur + vr;
                    imag[i + j] = ui + vi;
                    real[i + j + len / 2] = ur - vr;
                    imag[i + j + len / 2] = ui - vi;
                    float ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = ncr;
                }
            }
        }
    }

    private float[] autocorrelation(float[] x) {
        int n = x.length;
        float[] result = new float[n];

        for (int lag = 0; lag < n; lag++) {
            float sum = 0f;
            for (int i = 0; i < n - lag; i++) {
                sum += x[i] * x[i + lag];
            }
            result[lag] = sum;
        }

        return result;
    }
}
