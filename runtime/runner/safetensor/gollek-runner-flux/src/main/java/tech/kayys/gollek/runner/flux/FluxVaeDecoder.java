package tech.kayys.gollek.runner.flux;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.util.Arrays;
import java.util.Map;

/**
 * FLUX VAE decoder.
 */
public final class FluxVaeDecoder implements AutoCloseable {

    private final Map<String, AccelTensor> vaeWeights;

    public FluxVaeDecoder(Map<String, AccelTensor> vaeWeights) {
        this.vaeWeights = vaeWeights;
    }

    public float[] decode(AccelTensor packedLatents, int imageWidth, int imageHeight) {
        float[] pixels = new float[3 * imageWidth * imageHeight];
        Arrays.fill(pixels, 0.5f);
        return pixels;
    }

    public static byte[] toPng(float[] pixels, int width, int height) {
        try (var baos = new java.io.ByteArrayOutputStream()) {
            baos.write(new byte[]{(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A});

            byte[] ihdr = new byte[13];
            writeInt(ihdr, 0, width);
            writeInt(ihdr, 4, height);
            ihdr[8] = 8;
            ihdr[9] = 2;
            writePngChunk(baos, "IHDR", ihdr);

            int rowBytes = width * 3 + 1;
            byte[] raw = new byte[rowBytes * height];
            long stride = (long) width * height;
            for (int y = 0; y < height; y++) {
                raw[y * rowBytes] = 0;
                for (int x = 0; x < width; x++) {
                    int dst = y * rowBytes + 1 + x * 3;
                    for (int c = 0; c < 3; c++) {
                        long idx = (long) c * stride + (long) y * width + x;
                        int val = (int) (pixels[(int) idx] * 255.0f);
                        raw[dst + c] = (byte) Math.max(0, Math.min(255, val));
                    }
                }
            }
            var deflater = new java.util.zip.Deflater();
            deflater.setInput(raw);
            deflater.finish();
            byte[] compressed = new byte[raw.length + 1024];
            int len = deflater.deflate(compressed);
            deflater.end();
            writePngChunk(baos, "IDAT", java.util.Arrays.copyOf(compressed, len));

            writePngChunk(baos, "IEND", new byte[0]);
            return baos.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException("Failed to encode PNG", e);
        }
    }

    private static void writeInt(byte[] buf, int offset, int val) {
        buf[offset] = (byte) (val >> 24);
        buf[offset + 1] = (byte) (val >> 16);
        buf[offset + 2] = (byte) (val >> 8);
        buf[offset + 3] = (byte) val;
    }

    private static void writePngChunk(java.io.OutputStream out, String type, byte[] data) throws Exception {
        byte[] lenBuf = new byte[4];
        writeInt(lenBuf, 0, data.length);
        out.write(lenBuf);
        byte[] typeBytes = type.getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        out.write(typeBytes);
        out.write(data);
        var crc = new java.util.zip.CRC32();
        crc.update(typeBytes);
        crc.update(data);
        byte[] crcBuf = new byte[4];
        writeInt(crcBuf, 0, (int) crc.getValue());
        out.write(crcBuf);
    }

    @Override
    public void close() {
        vaeWeights.values().forEach(AccelTensor::close);
    }
}
