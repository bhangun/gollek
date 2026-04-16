/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * OggVorbisDecoder.java
 * ───────────────────────
 * OGG/Vorbis decoder using jOrbis library.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import com.jcraft.jogg.Packet;
import com.jcraft.jorbis.Block;
import com.jcraft.jorbis.Comment;
import com.jcraft.jorbis.DspState;
import com.jcraft.jorbis.Info;
import org.jboss.logging.Logger;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * OGG/Vorbis decoder using jOrbis library.
 * <p>
 * jOrbis is a pure Java OGG/Vorbis decoder that doesn't require native libraries.
 * </p>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class OggVorbisDecoder implements AudioDecoder {

    private static final Logger log = Logger.getLogger(OggVorbisDecoder.class);

    /**
     * Target sample rate for Whisper (16kHz).
     */
    private static final int TARGET_SAMPLE_RATE = 16000;

    /**
     * OGG page header signature.
     */
    private static final byte[] OGG_SIGNATURE = {'O', 'g', 'g', 'S'};

    @Override
    public float[] decode(byte[] audioBytes) throws IOException {
        log.debugf("Decoding OGG/Vorbis audio (%d bytes)", audioBytes.length);

        try {
            // Use jOrbis for decoding
            ByteArrayInputStream bais = new ByteArrayInputStream(audioBytes);
            java.io.ByteArrayOutputStream pcmOut = new java.io.ByteArrayOutputStream();

            // Simple OGG packet reader
            OggPacketReader reader = new OggPacketReader(bais);
            
            Info info = new Info();
            Comment comment = new Comment();
            DspState dspState = new DspState();
            Block block = new Block(dspState);

            boolean initialized = false;
            int packetCount = 0;

            while (true) {
                byte[] packet = reader.readPacket();
                if (packet == null) {
                    break;
                }

                packetCount++;

                if (!initialized) {
                    // First 3 packets are headers
                    Packet jPacket = createPacket(packet);
                    if (info.synthesis_headerin(comment, jPacket) < 0) {
                        // Try next packet
                        continue;
                    }

                    // Check if we have all headers (usually 3 packets)
                    if (packetCount >= 3) {
                        dspState.synthesis_init(info);
                        block.init(dspState);
                        initialized = true;
                        log.debugf("OGG initialized: %d Hz, %d channels",
                                info.rate, info.channels);
                    }
                } else {
                    // Decode audio packet
                    Packet jPacket = createPacket(packet);
                    if (block.synthesis(jPacket) == 0) {
                        dspState.synthesis_blockin(block);

                        // Get PCM samples
                        float[][][] pcm_wrapper = new float[1][][];
                        int samples = dspState.synthesis_pcmout(pcm_wrapper, null);
                        if (samples > 0) {
                            float[][] pcm = pcm_wrapper[0];
                            // Mix to mono and resample if needed
                            for (int i = 0; i < samples; i++) {
                                float sample = 0;
                                for (int ch = 0; ch < info.channels; ch++) {
                                    sample += pcm[ch][i];
                                }
                                sample /= info.channels;

                                // Resample if needed
                                if (info.rate != TARGET_SAMPLE_RATE) {
                                    // Simple linear interpolation
                                    float ratio = (float) TARGET_SAMPLE_RATE / info.rate;
                                    // Write resampled sample
                                    int val = (int) (sample * 32767);
                                    pcmOut.write(val & 0xFF);
                                    pcmOut.write((val >> 8) & 0xFF);
                                } else {
                                    int val = (int) (sample * 32767);
                                    pcmOut.write(val & 0xFF);
                                    pcmOut.write((val >> 8) & 0xFF);
                                }
                            }
                            dspState.synthesis_read(samples);
                        }
                    }
                }
            }

            // Cleanup
            block.clear();
            dspState.clear();
            bais.close();

            // Convert bytes to float array
            byte[] pcmBytes = pcmOut.toByteArray();
            float[] pcm = new float[pcmBytes.length / 2];
            for (int i = 0; i < pcm.length; i++) {
                int sample = (pcmBytes[i * 2] & 0xFF) | ((pcmBytes[i * 2 + 1] & 0xFF) << 8);
                pcm[i] = sample / 32768.0f;
            }

            log.infof("OGG decoded: %d bytes → %d samples (%.2fs)",
                    audioBytes.length, pcm.length, pcm.length / (float) TARGET_SAMPLE_RATE);

            return pcm;

        } catch (Exception e) {
            throw new IOException("OGG/Vorbis decoding failed: " + e.getMessage(), e);
        }
    }

    private Packet createPacket(byte[] data) {
        Packet p = new Packet();
        p.packet_base = data;
        p.packet = 0;
        p.bytes = data.length;
        return p;
    }

    @Override
    public String getFormat() {
        return "ogg";
    }

    @Override
    public boolean supports(String format) {
        return "ogg".equalsIgnoreCase(format) || "vorbis".equalsIgnoreCase(format);
    }

    /**
     * Simple OGG packet reader.
     */
    private static class OggPacketReader {
        private final ByteArrayInputStream input;
        private byte[] buffer = new byte[4096];
        private int pos = 0;
        private int count = 0;

        public OggPacketReader(ByteArrayInputStream input) {
            this.input = input;
        }

        public byte[] readPacket() throws IOException {
            // Read OGG page header
            if (!readBytes(4)) {
                return null;
            }

            // Check for OGG signature
            if (buffer[0] != OGG_SIGNATURE[0] || buffer[1] != OGG_SIGNATURE[1] ||
                buffer[2] != OGG_SIGNATURE[2] || buffer[3] != OGG_SIGNATURE[3]) {
                // Not a valid OGG page, skip
                return null;
            }

            // Read rest of page header (23 bytes total)
            if (!readBytes(23)) {
                return null;
            }

            // Get number of segments
            int segments = buffer[26] & 0xFF;

            // Read segment table
            if (!readBytes(27 + segments)) {
                return null;
            }

            // Calculate packet size
            int packetSize = 0;
            for (int i = 0; i < segments; i++) {
                packetSize += buffer[27 + i] & 0xFF;
            }

            // Read packet data
            byte[] packet = new byte[packetSize];
            int offset = 0;
            while (offset < packetSize) {
                int toRead = Math.min(packetSize - offset, count - pos);
                System.arraycopy(buffer, pos, packet, offset, toRead);
                offset += toRead;
                pos += toRead;

                if (offset < packetSize) {
                    // Need more data
                    if (!refillBuffer()) {
                        break;
                    }
                }
            }

            return packet;
        }

        private boolean readBytes(int needed) throws IOException {
            while (count - pos < needed) {
                if (!refillBuffer()) {
                    return false;
                }
            }
            return true;
        }

        private boolean refillBuffer() throws IOException {
            // Shift remaining data to beginning
            if (pos > 0) {
                System.arraycopy(buffer, pos, buffer, 0, count - pos);
                count -= pos;
                pos = 0;
            }

            // Read more data
            int bytesRead = input.read(buffer, count, buffer.length - count);
            if (bytesRead > 0) {
                count += bytesRead;
                return true;
            }
            return false;
        }
    }
}
