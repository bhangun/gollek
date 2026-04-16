/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioDecoderRegistry.java
 * ───────────────────────
 * Registry for audio decoders.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry and factory for audio decoders.
 * <p>
 * Provides unified access to all supported audio format decoders.
 * </p>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class AudioDecoderRegistry {

    private static final Logger log = Logger.getLogger(AudioDecoderRegistry.class);

    /**
     * Singleton instance.
     */
    private static final AudioDecoderRegistry INSTANCE = new AudioDecoderRegistry();

    /**
     * Registered decoders.
     */
    private final ConcurrentHashMap<String, AudioDecoder> decoders = new ConcurrentHashMap<>();

    /**
     * WAV decoder (built-in).
     */
    private final WavDecoder wavDecoder = new WavDecoder();

    /**
     * Private constructor for singleton.
     */
    private AudioDecoderRegistry() {
        // Register built-in decoders
        registerDecoder(wavDecoder);
        registerDecoder(new Mp3Decoder());
        registerDecoder(new FlacDecoder());
        registerDecoder(new OggVorbisDecoder());

        log.info("Audio decoder registry initialized with " + decoders.size() + " decoders");
    }

    /**
     * Get the singleton instance.
     *
     * @return registry instance
     */
    public static AudioDecoderRegistry getInstance() {
        return INSTANCE;
    }

    /**
     * Register an audio decoder.
     *
     * @param decoder decoder to register
     */
    public void registerDecoder(AudioDecoder decoder) {
        String format = decoder.getFormat().toLowerCase();
        decoders.put(format, decoder);
        log.debugf("Registered decoder for format: %s", format);
    }

    /**
     * Get a decoder for the specified format.
     *
     * @param format audio format (e.g., "mp3", "flac", "ogg")
     * @return decoder instance or null if not found
     */
    public AudioDecoder getDecoder(String format) {
        if (format == null) {
            return wavDecoder;
        }

        String normalizedFormat = format.toLowerCase().trim();

        // Handle common aliases
        normalizedFormat = normalizeFormat(normalizedFormat);

        AudioDecoder decoder = decoders.get(normalizedFormat);
        if (decoder == null) {
            log.warnf("No decoder found for format: %s, falling back to WAV decoder", format);
            return wavDecoder;
        }

        return decoder;
    }

    /**
     * Decode audio bytes to PCM float array.
     *
     * @param audioBytes input audio bytes
     * @param format     audio format
     * @return PCM float array normalized to [-1, 1]
     * @throws IOException if decoding fails
     */
    public float[] decode(byte[] audioBytes, String format) throws IOException {
        AudioDecoder decoder = getDecoder(format);
        return decoder.decode(audioBytes);
    }

    /**
     * Decode audio bytes using auto-detection of format.
     *
     * @param audioBytes input audio bytes
     * @return PCM float array normalized to [-1, 1]
     * @throws IOException if decoding fails
     */
    public float[] decode(byte[] audioBytes) throws IOException {
        String format = detectFormat(audioBytes);
        return decode(audioBytes, format);
    }

    /**
     * Get all supported formats.
     *
     * @return array of supported format names
     */
    public String[] getSupportedFormats() {
        return decoders.keySet().toArray(new String[0]);
    }

    /**
     * Check if a format is supported.
     *
     * @param format format name
     * @return true if supported
     */
    public boolean isSupported(String format) {
        if (format == null) {
            return true; // WAV is always supported
        }
        String normalized = normalizeFormat(format.toLowerCase().trim());
        return decoders.containsKey(normalized);
    }

    /**
     * Normalize format name to handle aliases.
     *
     * @param format format name
     * @return normalized format name
     */
    private String normalizeFormat(String format) {
        return switch (format) {
            case "mp3", "mpeg", "mpga" -> "mp3";
            case "flac", "fla" -> "flac";
            case "ogg", "vorbis", "oga" -> "ogg";
            case "wav", "wave", "rifx" -> "wav";
            case "m4a", "aac", "alac" -> "m4a";
            case "webm", "opus" -> "webm";
            default -> format;
        };
    }

    /**
     * Detect audio format from magic bytes.
     *
     * @param audioBytes audio bytes
     * @return detected format or "wav" as default
     */
    private String detectFormat(byte[] audioBytes) {
        if (audioBytes.length < 12) {
            return "wav";
        }

        // Check for WAV header (RIFF)
        if (audioBytes[0] == 'R' && audioBytes[1] == 'I' &&
                audioBytes[2] == 'F' && audioBytes[3] == 'F' &&
                audioBytes[8] == 'W' && audioBytes[9] == 'A' &&
                audioBytes[10] == 'V' && audioBytes[11] == 'E') {
            return "wav";
        }

        // Check for MP3 header (ID3 or frame sync)
        if ((audioBytes[0] == 'I' && audioBytes[1] == 'D' && audioBytes[2] == '3') ||
                (audioBytes[0] == (byte) 0xFF && (audioBytes[1] & 0xE0) == 0xE0)) {
            return "mp3";
        }

        // Check for FLAC header (fLaC)
        if (audioBytes[0] == 'f' && audioBytes[1] == 'L' &&
                audioBytes[2] == 'a' && audioBytes[3] == 'C') {
            return "flac";
        }

        // Check for OGG header (OggS)
        if (audioBytes[0] == 'O' && audioBytes[1] == 'g' &&
                audioBytes[2] == 'g' && audioBytes[3] == 'S') {
            return "ogg";
        }

        // Check for M4A header (ftyp)
        if (audioBytes[4] == 'f' && audioBytes[5] == 't' &&
                audioBytes[6] == 'y' && audioBytes[7] == 'p') {
            return "m4a";
        }

        // Default to WAV
        log.warn("Unknown audio format, assuming WAV");
        return "wav";
    }

    /**
     * Simple WAV decoder (built-in, no external dependencies).
     */
    private static class WavDecoder implements AudioDecoder {

        private static final int TARGET_SAMPLE_RATE = 16000;

        @Override
        public float[] decode(byte[] audioBytes) throws IOException {
            // Parse WAV header
            if (audioBytes.length < 44) {
                throw new IOException("WAV file too small");
            }

            // Skip to data chunk
            int dataOffset = 44;
            int dataLength = audioBytes.length - 44;

            // Read format info from header
            int channels = (audioBytes[22] & 0xFF) | ((audioBytes[23] & 0xFF) << 8);
            int sampleRate = (audioBytes[24] & 0xFF) | ((audioBytes[25] & 0xFF) << 8) |
                    ((audioBytes[26] & 0xFF) << 16) | ((audioBytes[27] & 0xFF) << 24);
            int bitsPerSample = (audioBytes[34] & 0xFF) | ((audioBytes[35] & 0xFF) << 8);
            int bytesPerSample = bitsPerSample / 8;

            // Find data chunk
            dataOffset = findDataChunk(audioBytes);
            if (dataOffset < 0) {
                throw new IOException("WAV data chunk not found");
            }
            dataLength = audioBytes.length - dataOffset;

            // Convert to float array
            int samples = dataLength / bytesPerSample / channels;
            float[] pcm = new float[samples];

            for (int i = 0; i < samples; i++) {
                float sample = 0;
                for (int ch = 0; ch < channels; ch++) {
                    int offset = dataOffset + (i * channels + ch) * bytesPerSample;
                    if (bitsPerSample == 16) {
                        short s = (short) ((audioBytes[offset] & 0xFF) |
                                ((audioBytes[offset + 1] & 0xFF) << 8));
                        sample += s / 32768.0f;
                    } else if (bitsPerSample == 24) {
                        int s = (audioBytes[offset] & 0xFF) |
                                ((audioBytes[offset + 1] & 0xFF) << 8) |
                                ((audioBytes[offset + 2] & 0xFF) << 16);
                        sample += s / 8388608.0f;
                    } else if (bitsPerSample == 32) {
                        int s = (audioBytes[offset] & 0xFF) |
                                ((audioBytes[offset + 1] & 0xFF) << 8) |
                                ((audioBytes[offset + 2] & 0xFF) << 16) |
                                ((audioBytes[offset + 3] & 0xFF) << 24);
                        sample += s / 2147483648.0f;
                    }
                }
                pcm[i] = sample / channels;
            }

            // Resample if needed
            if (sampleRate != TARGET_SAMPLE_RATE) {
                AudioResampler resampler = new AudioResampler(sampleRate, TARGET_SAMPLE_RATE);
                return resampler.resample(pcm);
            }

            return pcm;
        }

        private int findDataChunk(byte[] wavBytes) {
            // Search for "data" chunk marker
            for (int i = 0; i < wavBytes.length - 4; i++) {
                if (wavBytes[i] == 'd' && wavBytes[i + 1] == 'a' &&
                        wavBytes[i + 2] == 't' && wavBytes[i + 3] == 'a') {
                    // Found data chunk, skip 4 bytes (chunk ID) + 4 bytes (chunk size)
                    return i + 8;
                }
            }
            return -1;
        }

        @Override
        public String getFormat() {
            return "wav";
        }

        @Override
        public boolean supports(String format) {
            return "wav".equalsIgnoreCase(format);
        }
    }
}
