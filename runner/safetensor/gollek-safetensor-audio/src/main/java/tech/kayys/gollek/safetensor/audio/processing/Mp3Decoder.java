/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * Mp3Decoder.java
 * ───────────────────────
 * MP3 decoder using JLayer library.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import javazoom.jl.decoder.Bitstream;
import javazoom.jl.decoder.BitstreamException;
import javazoom.jl.decoder.Decoder;
import javazoom.jl.decoder.Header;
import javazoom.jl.decoder.JavaLayerException;
import javazoom.jl.decoder.SampleBuffer;
import org.jboss.logging.Logger;

import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * MP3 decoder using JLayer library.
 * <p>
 * JLayer is a pure Java MP3 decoder that doesn't require native libraries.
 * </p>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class Mp3Decoder implements AudioDecoder {

    private static final Logger log = Logger.getLogger(Mp3Decoder.class);

    /**
     * Target sample rate for Whisper (16kHz).
     */
    private static final int TARGET_SAMPLE_RATE = 16000;

    @Override
    public float[] decode(byte[] audioBytes) throws IOException {
        log.debugf("Decoding MP3 audio (%d bytes)", audioBytes.length);

        try {
            ByteArrayInputStream bais = new ByteArrayInputStream(audioBytes);
            Bitstream bitstream = new Bitstream(bais);
            Decoder decoder = new Decoder();

            // Use ByteArrayOutputStream-like approach to collect PCM samples
            java.io.ByteArrayOutputStream pcmOut = new java.io.ByteArrayOutputStream();
            int frameCount = 0;

            try {
                while (true) {
                    Header header = bitstream.readFrame();
                    if (header == null) {
                        break;
                    }

                    // Decode frame
                    SampleBuffer sampleBuffer = (SampleBuffer) decoder.decodeFrame(header, bitstream);
                    short[] samples = sampleBuffer.getBuffer();
                    int sampleRate = header.frequency();
                    int channels = sampleBuffer.getChannelCount();

                    // Resample if needed
                    float[] resampled = resampleIfNeeded(samples, sampleRate, channels);

                    // Write to output
                    for (float sample : resampled) {
                        // Convert float [-1,1] to bytes
                        int val = (int) (sample * 32767);
                        pcmOut.write(val & 0xFF);
                        pcmOut.write((val >> 8) & 0xFF);
                    }

                    frameCount++;
                    bitstream.closeFrame();
                }
            } catch (BitstreamException e) {
                if (!"No more frames".equals(e.getMessage())) {
                    throw e;
                }
                // End of stream - normal exit
            }

            bitstream.close();
            log.debugf("Decoded %d MP3 frames", frameCount);

            // Convert bytes back to float array
            byte[] pcmBytes = pcmOut.toByteArray();
            float[] pcm = new float[pcmBytes.length / 2];
            for (int i = 0; i < pcm.length; i++) {
                int sample = (pcmBytes[i * 2] & 0xFF) | ((pcmBytes[i * 2 + 1] & 0xFF) << 8);
                pcm[i] = sample / 32768.0f;
            }

            log.infof("MP3 decoded: %d bytes → %d samples (%.2fs)",
                    audioBytes.length, pcm.length, pcm.length / (float) TARGET_SAMPLE_RATE);

            return pcm;

        } catch (JavaLayerException e) {
            throw new IOException("MP3 decoding failed: " + e.getMessage(), e);
        }
    }

    @Override
    public String getFormat() {
        return "mp3";
    }

    @Override
    public boolean supports(String format) {
        return "mp3".equalsIgnoreCase(format) || "mpeg".equalsIgnoreCase(format);
    }

    /**
     * Resample audio to target sample rate if needed.
     *
     * @param samples    input samples
     * @param sourceRate source sample rate
     * @param channels   number of channels
     * @return resampled mono float array
     */
    private float[] resampleIfNeeded(short[] samples, int sourceRate, int channels) {
        // Convert to mono if stereo
        float[] mono;
        if (channels == 2) {
            mono = new float[samples.length / 2];
            for (int i = 0, j = 0; i < mono.length; i++, j += 2) {
                mono[i] = (samples[j] + samples[j + 1]) / 2.0f / 32768.0f;
            }
        } else {
            mono = new float[samples.length];
            for (int i = 0; i < samples.length; i++) {
                mono[i] = samples[i] / 32768.0f;
            }
        }

        // Resample if sample rate doesn't match
        if (sourceRate != TARGET_SAMPLE_RATE) {
            AudioResampler resampler = new AudioResampler(sourceRate, TARGET_SAMPLE_RATE);
            return resampler.resample(mono);
        }

        return mono;
    }
}
