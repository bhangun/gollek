package tech.kayys.gollek.safetensor.audio.processing;

import de.sciss.jump3r.lowlevel.LameEncoder;
import org.jboss.logging.Logger;

import javax.sound.sampled.AudioFormat;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Pure Java MP3 encoder using jump3r (LAME port).
 */
public class Mp3Encoder {
    private static final Logger LOG = Logger.getLogger(Mp3Encoder.class);

    private final int sampleRate;
    private final int channels;
    private final int bitRate;

    public Mp3Encoder(int sampleRate, int channels, int bitRate) {
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.bitRate = bitRate;
    }

    public Mp3Encoder(int sampleRate) {
        this(sampleRate, 1, 128); // Default to mono 128kbps
    }

    /**
     * Encode PCM float array (-1.0 to 1.0) to MP3 bytes.
     */
    public byte[] encode(float[] pcm) throws IOException {
        // Convert float[] to short[] (16-bit PCM)
        short[] shortPcm = new short[pcm.length];
        for (int i = 0; i < pcm.length; i++) {
            float val = pcm[i] * 32767.0f;
            if (val > 32767.0f) val = 32767.0f;
            if (val < -32768.0f) val = -32768.0f;
            shortPcm[i] = (short) val;
        }

        // Initialize LAME encoder
        // Note: LameEncoder constructor parameters:
        // (sourceFormat, bitRate, mode, quality, variableBitRate)
        AudioFormat sourceFormat = new AudioFormat(sampleRate, 16, channels, true, false);
        LameEncoder encoder = new LameEncoder(sourceFormat, bitRate, LameEncoder.CHANNEL_MODE_MONO, 2, false);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[encoder.getMP3BufferSize()];

        // Encode in chunks
        int bytesPerSample = 2; // 16-bit
        byte[] pcmBytes = new byte[shortPcm.length * bytesPerSample];
        ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(shortPcm);

        int offset = 0;
        while (offset < pcmBytes.length) {
            int toRead = Math.min(pcmBytes.length - offset, 1024 * 16);
            int encoded = encoder.encodeBuffer(pcmBytes, offset, toRead, buffer);
            if (encoded > 0) {
                out.write(buffer, 0, encoded);
            }
            offset += toRead;
        }

        // Flush encoder
        int finalEncoded = encoder.encodeFinish(buffer);
        if (finalEncoded > 0) {
            out.write(buffer, 0, finalEncoded);
        }

        encoder.close();
        return out.toByteArray();
    }
}
