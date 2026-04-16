package tech.kayys.gollek.converter.gguf;

import java.io.IOException;
import java.nio.file.*;
import java.util.List;

/**
 * Simple self-contained integration tests (no JUnit dependency).
 * Run via: {@code java -cp gguf-converter.jar io.gguf.GgufSelfTest}
 */
public final class GgufSelfTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) throws Exception {
        testF16Conversion();
        testBf16Conversion();
        testQ8_0Quantization();
        testQ4_0Quantization();
        testGgufRoundTrip();
        testMetadataTypes();

        System.out.printf("%n=== Results: %d passed, %d failed ===%n", passed, failed);
        if (failed > 0)
            System.exit(1);
    }

    // ── F16 conversion ────────────────────────────────────────────────────

    static void testF16Conversion() {
        float[] vals = { 0f, 1f, -1f, 3.14f, Float.MAX_VALUE / 2, -0.001f };
        byte[] f32 = floatsToBytes(vals);
        byte[] f16 = TensorConverter.f32ToF16(f32, vals.length);
        byte[] back = TensorConverter.f16ToF32(f16, vals.length);
        float[] result = bytesToFloats(back, vals.length);

        for (int i = 0; i < vals.length; i++) {
            float rel = Math.abs(vals[i]) > 1e-6f
                    ? Math.abs((result[i] - vals[i]) / vals[i])
                    : Math.abs(result[i] - vals[i]);
            assert_("F16 round-trip[" + i + "] rel=" + rel, rel < 0.01f);
        }
        pass("F16 conversion round-trip");
    }

    // ── BF16 conversion ───────────────────────────────────────────────────

    static void testBf16Conversion() {
        // Manually encode 1.0f in BF16: 0x3F80 (upper 16 bits of 0x3F800000)
        byte[] bf16 = { (byte) 0x80, 0x3F }; // little-endian 0x3F80
        byte[] f32 = TensorConverter.bf16ToF32(bf16, 1);
        float val = java.nio.ByteBuffer.wrap(f32).order(java.nio.ByteOrder.LITTLE_ENDIAN).getFloat();
        assert_("BF16 1.0f", Math.abs(val - 1.0f) < 1e-6f);
        pass("BF16 → F32 conversion");
    }

    // ── Q8_0 quantization ─────────────────────────────────────────────────

    static void testQ8_0Quantization() {
        // 32 elements all equal to 1.0f
        float[] vals = new float[32];
        java.util.Arrays.fill(vals, 1.0f);
        byte[] f32 = floatsToBytes(vals);
        byte[] q8 = TensorConverter.quantizeQ8_0(f32, 32);
        assert_("Q8_0 block size", q8.length == 34);

        // Scale should be 1/127 → stored as F16 ≈ 0x2800 area
        // All weights should be 127
        for (int i = 2; i < 34; i++) {
            assert_("Q8_0 weight[" + (i - 2) + "]=127", q8[i] == (byte) 127);
        }
        pass("Q8_0 quantization");
    }

    // ── Q4_0 quantization ─────────────────────────────────────────────────

    static void testQ4_0Quantization() {
        float[] vals = new float[32];
        for (int i = 0; i < 32; i++)
            vals[i] = (i % 2 == 0) ? 7f : -7f;
        byte[] f32 = floatsToBytes(vals);
        byte[] q4 = TensorConverter.quantizeQ4_0(f32, 32);
        assert_("Q4_0 block size", q4.length == 18);
        pass("Q4_0 quantization");
    }

    // ── GGUF write + read round-trip ──────────────────────────────────────

    static void testGgufRoundTrip() throws IOException {
        GgufModel src = new GgufModel();
        src.addMeta("general.architecture", GgufMetaValue.ofString("llama"));
        src.addMeta("general.name", GgufMetaValue.ofString("test-model"));
        src.addMeta("llama.block_count", GgufMetaValue.ofUInt32(2));
        src.addMeta("llama.embedding_length", GgufMetaValue.ofUInt32(64));

        // One small F32 tensor: shape [4, 8] = 32 elements
        float[] weights = new float[32];
        for (int i = 0; i < 32; i++)
            weights[i] = i * 0.01f;
        byte[] tensorBytes = floatsToBytes(weights);

        long[] ne = { 8L, 4L }; // innermost first
        src.addTensor(new TensorInfo("token_embd.weight", ne, GgmlType.F32, 0));
        // Pad to alignment (32 bytes already aligned for 128-byte blob)
        byte[] blob = java.util.Arrays.copyOf(tensorBytes, 128);
        src.setTensorData(blob);

        Path tmp = Files.createTempFile("gguf-test-", ".gguf");
        try {
            GgufWriter.write(src, tmp);

            try (GgufReader reader = new GgufReader(tmp)) {
                GgufModel dst = reader.read();

                assert_("RT arch", dst.architecture().equals("llama"));
                assert_("RT name", dst.modelName().equals("test-model"));
                assert_("RT tensors", dst.tensors().size() == 1);
                assert_("RT kv count", dst.metadata().size() == 4);

                TensorInfo ti = dst.tensors().get(0);
                assert_("RT tensor name", ti.name().equals("token_embd.weight"));
                assert_("RT tensor type", ti.type() == GgmlType.F32);
                assert_("RT tensor ne[0]", ti.ne()[0] == 8L);
                assert_("RT tensor ne[1]", ti.ne()[1] == 4L);
            }
            pass("GGUF write/read round-trip");
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    // ── Metadata type serialization ───────────────────────────────────────

    static void testMetadataTypes() throws IOException {
        GgufModel src = new GgufModel();
        src.addMeta("test.uint8", new GgufMetaValue.UInt8Val((short) 255));
        src.addMeta("test.int8", new GgufMetaValue.Int8Val((byte) -1));
        src.addMeta("test.uint32", GgufMetaValue.ofUInt32(0xFFFFFFFFL));
        src.addMeta("test.float32", GgufMetaValue.ofFloat32(3.14f));
        src.addMeta("test.bool", GgufMetaValue.ofBool(true));
        src.addMeta("test.string", GgufMetaValue.ofString("hello GGUF"));
        src.addMeta("test.strarr", GgufMetaValue.ofStringArray(List.of("a", "b", "c")));
        src.addMeta("test.int32arr", GgufMetaValue.ofInt32Array(List.of(1, 2, 3)));
        src.setTensorData(new byte[0]);

        Path tmp = Files.createTempFile("gguf-meta-", ".gguf");
        try {
            GgufWriter.write(src, tmp);
            try (GgufReader r = new GgufReader(tmp)) {
                GgufModel dst = r.read();
                assert_("RT uint8", ((GgufMetaValue.UInt8Val) dst.getMeta("test.uint8").orElseThrow()).value() == 255);
                assert_("RT float32", Math.abs(dst.getMeta("test.float32").orElseThrow().asFloat32() - 3.14f) < 0.001f);
                assert_("RT bool", dst.getMeta("test.bool").orElseThrow().asBool());
                assert_("RT string", dst.getMeta("test.string").orElseThrow().asString().equals("hello GGUF"));
                var arr = dst.getMeta("test.strarr").orElseThrow().asArray();
                assert_("RT strarr.size", arr.size() == 3);
                assert_("RT strarr[1]", arr.get(1).asString().equals("b"));
            }
            pass("Metadata type round-trip");
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static byte[] floatsToBytes(float[] vals) {
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(vals.length * 4)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float v : vals)
            bb.putFloat(v);
        return bb.array();
    }

    private static float[] bytesToFloats(byte[] bytes, int count) {
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(bytes)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[count];
        for (int i = 0; i < count; i++)
            out[i] = bb.getFloat();
        return out;
    }

    private static void assert_(String label, boolean cond) {
        if (!cond) {
            System.out.println("  ✗ FAIL: " + label);
            failed++;
        }
    }

    private static void pass(String label) {
        System.out.println("  ✓ PASS: " + label);
        passed++;
    }
}
