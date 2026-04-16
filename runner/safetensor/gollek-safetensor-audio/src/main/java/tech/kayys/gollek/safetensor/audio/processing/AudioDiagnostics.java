/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioDiagnostics.java
 * ───────────────────────
 * Audio analysis and diagnostic utilities.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Comprehensive audio analysis and diagnostic utilities.
 * 
 * <p>Provides detailed audio file analysis including format detection,
 * quality metrics, and visualization tools.</p>
 *
 * <h2>Features</h2>
 * <ul>
 *   <li>Audio file format detection</li>
 *   <li>Quality metrics (SNR, dynamic range, clipping)</li>
 *   <li>Format information extraction</li>
 *   <li>ASCII waveform visualization</li>
 *   <li>Statistical analysis</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Analyze audio file
 * AudioReport report = AudioDiagnostics.analyze(Path.of("audio.flac"));
 * System.out.println("Format: " + report.format());
 * System.out.println("Quality: " + report.qualityGrade());
 * 
 * // Visualize waveform
 * String waveform = AudioDiagnostics.waveformAscii(pcm, 80);
 * System.out.println(waveform);
 * 
 * // Quick quality check
 * if (report.isClipping()) {
 *     System.out.println("Warning: Audio is clipping!");
 * }
 * }</pre>
 *
 * @author Bhangun
 * @version 1.0.0
 * @since 2.1.0
 */
public final class AudioDiagnostics {

    private static final Logger log = Logger.getLogger(AudioDiagnostics.class);

    private AudioDiagnostics() {
        // Utility class - prevent instantiation
    }

    /**
     * Analyze an audio file and generate a comprehensive report.
     *
     * @param audioFile path to audio file
     * @return audio analysis report
     * @throws IOException if file cannot be read
     */
    public static AudioReport analyze(Path audioFile) throws IOException {
        if (!Files.exists(audioFile)) {
            throw new IOException("File does not exist: " + audioFile);
        }

        byte[] bytes = Files.readAllBytes(audioFile);
        String format = detectFormat(bytes);
        
        // Parse format-specific information
        Map<String, String> metadata = new HashMap<>();
        int sampleRate = 0;
        int channels = 0;
        int bitsPerSample = 0;
        long durationMs = 0;

        if ("flac".equals(format)) {
            // Parse FLAC header (simplified)
            if (bytes.length > 4 && bytes[0] == 'f' && bytes[1] == 'L' && 
                bytes[2] == 'a' && bytes[3] == 'C') {
                metadata.put("codec", "FLAC");
                metadata.put("container", "FLAC");
                // Would need full parsing for detailed info
            }
        } else if ("wav".equals(format)) {
            // Parse WAV header
            if (bytes.length > 44) {
                channels = (bytes[22] & 0xFF) | ((bytes[23] & 0xFF) << 8);
                sampleRate = (bytes[24] & 0xFF) | ((bytes[25] & 0xFF) << 8) |
                            ((bytes[26] & 0xFF) << 16) | ((bytes[27] & 0xFF) << 24);
                bitsPerSample = (bytes[34] & 0xFF) | ((bytes[35] & 0xFF) << 8);
                
                metadata.put("codec", "PCM");
                metadata.put("container", "WAV");
            }
        } else if ("mp3".equals(format)) {
            metadata.put("codec", "MP3");
            metadata.put("container", "MP3");
        }

        // Decode to PCM for quality analysis
        float[] pcm = decodeToPcm(bytes, format);
        
        // Calculate quality metrics
        float peakLevel = AudioProcessor.calculatePeak(pcm);
        float rmsLevel = AudioProcessor.calculateRMS(pcm);
        float clippingRatio = AudioProcessor.calculateClippingRatio(pcm, 0.99f);
        float zeroCrossingRate = AudioProcessor.calculateZeroCrossingRate(pcm);
        
        // Estimate SNR (simplified)
        float snr = estimateSNR(pcm);
        
        // Calculate dynamic range
        float dynamicRange = calculateDynamicRange(pcm);
        
        // Determine if lossy
        boolean isLossy = "mp3".equals(format) || "ogg".equals(format);
        
        // Calculate quality grade
        String qualityGrade = calculateQualityGrade(snr, dynamicRange, clippingRatio, isLossy);

        // Estimate duration
        if (sampleRate > 0 && pcm.length > 0) {
            durationMs = (pcm.length * 1000L) / sampleRate;
        }

        return new AudioReport(
            format,
            sampleRate,
            channels,
            bitsPerSample,
            durationMs,
            peakLevel,
            rmsLevel,
            clippingRatio > 0.01f, // More than 1% clipped
            isLossy,
            qualityGrade,
            snr,
            dynamicRange,
            zeroCrossingRate,
            metadata
        );
    }

    /**
     * Analyze PCM audio data.
     *
     * @param pcm PCM float array
     * @param sampleRate sample rate in Hz
     * @return audio analysis report
     */
    public static AudioReport analyze(float[] pcm, int sampleRate) {
        float peakLevel = AudioProcessor.calculatePeak(pcm);
        float rmsLevel = AudioProcessor.calculateRMS(pcm);
        float clippingRatio = AudioProcessor.calculateClippingRatio(pcm, 0.99f);
        float zeroCrossingRate = AudioProcessor.calculateZeroCrossingRate(pcm);
        float snr = estimateSNR(pcm);
        float dynamicRange = calculateDynamicRange(pcm);
        
        String qualityGrade = calculateQualityGrade(snr, dynamicRange, clippingRatio, false);
        long durationMs = (pcm.length * 1000L) / sampleRate;

        return new AudioReport(
            "pcm",
            sampleRate,
            1, // Assume mono
            32, // Float32
            durationMs,
            peakLevel,
            rmsLevel,
            clippingRatio > 0.01f,
            false,
            qualityGrade,
            snr,
            dynamicRange,
            zeroCrossingRate,
            Map.of()
        );
    }

    /**
     * Detect audio format from magic bytes.
     *
     * @param bytes audio bytes
     * @return detected format
     */
    public static String detectFormat(byte[] bytes) {
        if (bytes.length < 12) {
            return "unknown";
        }

        // FLAC: fLaC
        if (bytes[0] == 'f' && bytes[1] == 'L' && 
            bytes[2] == 'a' && bytes[3] == 'C') {
            return "flac";
        }

        // WAV: RIFF....WAVE
        if (bytes[0] == 'R' && bytes[1] == 'I' && bytes[2] == 'F' && bytes[3] == 'F' &&
            bytes[8] == 'W' && bytes[9] == 'A' && bytes[10] == 'V' && bytes[11] == 'E') {
            return "wav";
        }

        // MP3: ID3 or frame sync
        if ((bytes[0] == 'I' && bytes[1] == 'D' && bytes[2] == '3') ||
            (bytes[0] == (byte) 0xFF && (bytes[1] & 0xE0) == 0xE0)) {
            return "mp3";
        }

        // OGG: OggS
        if (bytes[0] == 'O' && bytes[1] == 'g' && bytes[2] == 'g' && bytes[3] == 'S') {
            return "ogg";
        }

        // M4A: ....ftyp
        if (bytes[4] == 'f' && bytes[5] == 't' && bytes[6] == 'y' && bytes[7] == 'p') {
            return "m4a";
        }

        return "unknown";
    }

    /**
     * Generate ASCII waveform visualization.
     *
     * @param pcm PCM float array
     * @param width terminal width in characters
     * @return ASCII waveform
     */
    public static String waveformAscii(float[] pcm, int width) {
        if (pcm == null || pcm.length == 0 || width < 10) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        int step = Math.max(1, pcm.length / width);
        char[] chars = new char[width];

        for (int i = 0; i < width; i++) {
            int idx = Math.min(i * step, pcm.length - 1);
            float sample = pcm[idx];
            
            // Map sample (-1 to 1) to characters
            if (sample > 0.9f) {
                chars[i] = '█';
            } else if (sample > 0.7f) {
                chars[i] = '▓';
            } else if (sample > 0.5f) {
                chars[i] = '▒';
            } else if (sample > 0.3f) {
                chars[i] = '░';
            } else if (sample > 0.1f) {
                chars[i] = '·';
            } else if (sample > -0.1f) {
                chars[i] = '─';
            } else if (sample > -0.3f) {
                chars[i] = '░';
            } else if (sample > -0.5f) {
                chars[i] = '▒';
            } else if (sample > -0.7f) {
                chars[i] = '▓';
            } else {
                chars[i] = '█';
            }
        }

        sb.append(chars);
        return sb.toString();
    }

    /**
     * Generate detailed waveform with amplitude markers.
     *
     * @param pcm PCM float array
     * @param width terminal width
     * @param height terminal height (lines)
     * @return multi-line ASCII waveform
     */
    public static String waveformAsciiDetailed(float[] pcm, int width, int height) {
        if (pcm == null || pcm.length == 0) {
            return "";
        }

        char[][] canvas = new char[height][width];
        int midRow = height / 2;

        // Initialize canvas
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                canvas[row][col] = row == midRow ? '─' : ' ';
            }
        }

        int step = Math.max(1, pcm.length / width);
        for (int col = 0; col < width; col++) {
            int idx = Math.min(col * step, pcm.length - 1);
            float sample = pcm[idx];
            
            // Map sample to row
            int rowOffset = (int) (sample * (height / 2));
            int row = midRow - rowOffset;
            
            if (row >= 0 && row < height) {
                canvas[row][col] = '│';
            }
        }

        StringBuilder sb = new StringBuilder();
        for (char[] row : canvas) {
            sb.append(row).append('\n');
        }
        return sb.toString();
    }

    /**
     * Decode audio bytes to PCM.
     *
     * @param bytes audio bytes
     * @param format audio format
     * @return PCM float array
     */
    private static float[] decodeToPcm(byte[] bytes, String format) {
        try {
            AudioDecoder decoder = switch (format) {
                case "flac" -> new FlacDecoder();
                case "wav" -> new WavDecoder();
                default -> new FlacDecoder(); // Fallback
            };
            return decoder.decode(bytes);
        } catch (IOException e) {
            log.warnf("Failed to decode audio: %s", e.getMessage());
            return new float[0];
        }
    }

    /**
     * Estimate signal-to-noise ratio.
     *
     * @param pcm PCM float array
     * @return estimated SNR in dB
     */
    private static float estimateSNR(float[] pcm) {
        if (pcm.length < 1000) {
            return 0;
        }

        // Find quiet sections (assumed to be noise floor)
        float noiseSum = 0;
        int noiseCount = 0;
        float threshold = 0.01f;

        for (float sample : pcm) {
            if (Math.abs(sample) < threshold) {
                noiseSum += sample * sample;
                noiseCount++;
            }
        }

        if (noiseCount < 10) {
            return 60; // Assume good SNR
        }

        float noiseRMS = (float) Math.sqrt(noiseSum / noiseCount);
        float signalRMS = AudioProcessor.calculateRMS(pcm);

        if (noiseRMS < 1e-6f) {
            return 90; // Very clean
        }

        return 20 * (float) Math.log10(signalRMS / noiseRMS);
    }

    /**
     * Calculate dynamic range.
     *
     * @param pcm PCM float array
     * @return dynamic range in dB
     */
    private static float calculateDynamicRange(float[] pcm) {
        float peak = AudioProcessor.calculatePeak(pcm);
        float rms = AudioProcessor.calculateRMS(pcm);

        if (peak < 1e-6f || rms < 1e-6f) {
            return 0;
        }

        return 20 * (float) Math.log10(peak / rms);
    }

    /**
     * Calculate quality grade.
     *
     * @param snr SNR in dB
     * @param dynamicRange dynamic range in dB
     * @param clippingRatio clipping ratio
     * @param isLossy whether audio is lossy
     * @return quality grade (A-F)
     */
    private static String calculateQualityGrade(float snr, float dynamicRange, 
                                                 float clippingRatio, boolean isLossy) {
        float score = 100;

        // SNR contribution
        if (snr < 20) score -= 30;
        else if (snr < 40) score -= 20;
        else if (snr < 60) score -= 10;

        // Dynamic range contribution
        if (dynamicRange < 5) score -= 20;
        else if (dynamicRange < 10) score -= 10;
        else if (dynamicRange < 15) score -= 5;

        // Clipping penalty
        if (clippingRatio > 0.1f) score -= 30;
        else if (clippingRatio > 0.05f) score -= 20;
        else if (clippingRatio > 0.01f) score -= 10;

        // Lossy penalty
        if (isLossy) score -= 10;

        if (score >= 90) return "A";
        if (score >= 80) return "B";
        if (score >= 70) return "C";
        if (score >= 60) return "D";
        return "F";
    }

    /**
     * Simple WAV decoder for diagnostics.
     */
    private static class WavDecoder implements AudioDecoder {
        @Override
        public float[] decode(byte[] audioBytes) throws IOException {
            if (audioBytes.length < 44) {
                throw new IOException("WAV file too small");
            }

            int channels = (audioBytes[22] & 0xFF) | ((audioBytes[23] & 0xFF) << 8);
            int bitsPerSample = (audioBytes[34] & 0xFF) | ((audioBytes[35] & 0xFF) << 8);
            int dataOffset = findDataChunk(audioBytes);
            
            if (dataOffset < 0) {
                throw new IOException("WAV data chunk not found");
            }

            int samples = (audioBytes.length - dataOffset) / (bitsPerSample / 8) / channels;
            float[] pcm = new float[samples];

            ByteBuffer buffer = ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN);
            buffer.position(dataOffset);

            for (int i = 0; i < samples; i++) {
                float sample = 0;
                for (int ch = 0; ch < channels; ch++) {
                    if (bitsPerSample == 16) {
                        short s = buffer.getShort();
                        sample += s / 32768.0f;
                    } else if (bitsPerSample == 32) {
                        int s = buffer.getInt();
                        sample += s / 2147483648.0f;
                    }
                }
                pcm[i] = sample / channels;
            }

            return pcm;
        }

        private int findDataChunk(byte[] wavBytes) {
            for (int i = 0; i < wavBytes.length - 4; i++) {
                if (wavBytes[i] == 'd' && wavBytes[i + 1] == 'a' &&
                    wavBytes[i + 2] == 't' && wavBytes[i + 3] == 'a') {
                    return i + 8;
                }
            }
            return -1;
        }

        @Override
        public String getFormat() { return "wav"; }
        @Override
        public boolean supports(String format) { return "wav".equalsIgnoreCase(format); }
    }

    /**
     * Audio analysis report.
     *
     * @param format audio format (flac, wav, mp3, etc.)
     * @param sampleRate sample rate in Hz
     * @param channels number of channels
     * @param bitsPerSample bits per sample
     * @param durationMs duration in milliseconds
     * @param peakLevel peak amplitude (0.0 to 1.0)
     * @param rmsLevel RMS level (0.0 to 1.0)
     * @param isClipping whether audio is clipping
     * @param isLossy whether audio is lossy compressed
     * @param qualityGrade quality grade (A-F)
     * @param snr signal-to-noise ratio in dB
     * @param dynamicRange dynamic range in dB
     * @param zeroCrossingRate zero-crossing rate
     * @param metadata additional metadata
     */
    public record AudioReport(
            String format,
            int sampleRate,
            int channels,
            int bitsPerSample,
            long durationMs,
            float peakLevel,
            float rmsLevel,
            boolean isClipping,
            boolean isLossy,
            String qualityGrade,
            float snr,
            float dynamicRange,
            float zeroCrossingRate,
            Map<String, String> metadata
    ) {
        /**
         * Get duration as formatted string.
         *
         * @return duration as "MM:SS.ms"
         */
        public String durationFormatted() {
            long minutes = durationMs / 60000;
            long seconds = (durationMs % 60000) / 1000;
            long ms = durationMs % 1000;
            return String.format("%02d:%02d.%03d", minutes, seconds, ms);
        }

        /**
         * Get bitrate estimate in kbps.
         *
         * @return bitrate in kbps
         */
        public int bitrateKbps() {
            if (durationMs == 0) return 0;
            // Simplified estimate
            return (sampleRate * channels * bitsPerSample) / 1000;
        }

        @Override
        public String toString() {
            return String.format(
                "AudioReport[format=%s, %dHz/%dch/%d-bit, %s, grade=%s, SNR=%.1fdB, DR=%.1fdB]",
                format, sampleRate, channels, bitsPerSample, 
                durationFormatted(), qualityGrade, snr, dynamicRange);
        }
    }
}
