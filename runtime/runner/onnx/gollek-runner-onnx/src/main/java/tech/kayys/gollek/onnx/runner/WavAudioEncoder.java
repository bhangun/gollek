package tech.kayys.gollek.onnx.runner;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Map;

final class WavAudioEncoder {
    private WavAudioEncoder() {
    }

    static byte[] encodePcm16(float[] channelMajorAudio, int channels, int samples, int sampleRate) {
        return encodePcm16(channelMajorAudio, channels, samples, sampleRate, Map.of());
    }

    static byte[] encodePcm16(
            float[] channelMajorAudio,
            int channels,
            int samples,
            int sampleRate,
            Map<String, String> infoMetadata) {
        int bytesPerSample = 2;
        int dataSize = samples * channels * bytesPerSample;
        byte[] infoChunk = buildInfoChunk(infoMetadata);
        int riffSize = 36 + infoChunk.length + dataSize;
        ByteArrayOutputStream out = new ByteArrayOutputStream(44 + infoChunk.length + dataSize);
        try {
            writeAscii(out, "RIFF");
            writeIntLE(out, riffSize);
            writeAscii(out, "WAVE");
            writeAscii(out, "fmt ");
            writeIntLE(out, 16);
            writeShortLE(out, (short) 1);
            writeShortLE(out, (short) channels);
            writeIntLE(out, sampleRate);
            writeIntLE(out, sampleRate * channels * bytesPerSample);
            writeShortLE(out, (short) (channels * bytesPerSample));
            writeShortLE(out, (short) 16);
            out.write(infoChunk);
            writeAscii(out, "data");
            writeIntLE(out, dataSize);

            for (int sample = 0; sample < samples; sample++) {
                for (int channel = 0; channel < channels; channel++) {
                    float value = channelMajorAudio[channel * samples + sample];
                    int pcm = Math.round(Math.max(-1.0f, Math.min(1.0f, value)) * 32767.0f);
                    writeShortLE(out, (short) pcm);
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to encode WAV audio", e);
        }
        return out.toByteArray();
    }

    private static byte[] buildInfoChunk(Map<String, String> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return new byte[0];
        }

        ByteArrayOutputStream info = new ByteArrayOutputStream();
        try {
            writeAscii(info, "INFO");
            for (Map.Entry<String, String> entry : metadata.entrySet()) {
                String key = normalizeInfoKey(entry.getKey());
                String value = normalizeInfoValue(entry.getValue());
                if (key.isBlank() || value.isBlank()) {
                    continue;
                }
                byte[] payload = value.getBytes(StandardCharsets.UTF_8);
                int chunkSize = payload.length + 1;
                writeAscii(info, key);
                writeIntLE(info, chunkSize);
                info.write(payload);
                info.write(0);
                if ((chunkSize & 1) != 0) {
                    info.write(0);
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to encode WAV INFO metadata", e);
        }

        if (info.size() <= 4) {
            return new byte[0];
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream(8 + info.size() + (info.size() & 1));
        try {
            writeAscii(out, "LIST");
            writeIntLE(out, info.size());
            out.write(info.toByteArray());
            if ((info.size() & 1) != 0) {
                out.write(0);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to encode WAV LIST metadata", e);
        }
        return out.toByteArray();
    }

    private static String normalizeInfoKey(String key) {
        if (key == null) {
            return "";
        }
        String value = key.trim().toUpperCase(java.util.Locale.ROOT);
        if (value.length() != 4) {
            return "";
        }
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (c < 'A' || c > 'Z') {
                return "";
            }
        }
        return value;
    }

    private static String normalizeInfoValue(String value) {
        if (value == null) {
            return "";
        }
        return value.strip().replace('\0', ' ');
    }

    private static void writeAscii(ByteArrayOutputStream out, String value) throws IOException {
        out.write(value.getBytes(StandardCharsets.US_ASCII));
    }

    private static void writeIntLE(ByteArrayOutputStream out, int value) throws IOException {
        out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array());
    }

    private static void writeShortLE(ByteArrayOutputStream out, short value) throws IOException {
        out.write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value).array());
    }
}
