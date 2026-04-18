package tech.kayys.gollek.quantizer.turboquant;

import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;
import tech.kayys.gollek.gguf.loader.GGUFDequantizer;

/**
 * Tests covering GGUF, BnB NF4, HQQ, SqueezeLLM, Registry, and infrastructure.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ExtendedQuantizerTest {

    // ═══════════════════════════════════════════════════════════════════════════
    // GGUF Tests
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @Order(1)
    @DisplayName("GGUF Q4_0: single block dequant (q=8 → w=0)")
    void testGgufQ4_0MidZero() {
        GGUFDequantizer dq = new GGUFDequantizer();
        // 1 block = 32 elements = 18 bytes
        // scale = 1.0 (FP16 = 0x3C00), all nibbles = 8 → q-8 = 0
        byte[] block = new byte[18];
        // FP16 1.0 = 0x3C00 → LE bytes [0x00, 0x3C]
        block[0] = 0x00;
        block[1] = 0x3C;
        // Nibbles: all 8 = 0x88 packed
        for (int i = 2; i < 18; i++)
            block[i] = (byte) 0x88;

        float[] out = new float[32];
        dq.dequantQ4_0(block, 32, out);
        for (float v : out)
            assertEquals(0.0f, v, 0.01f, "nibble=8 → q-8=0 → w=0");
    }

    @Test
    @Order(2)
    @DisplayName("GGUF Q4_0: extreme nibbles (0 → -8d, 15 → +7d)")
    void testGgufQ4_0Extremes() {
        GGUFDequantizer dq = new GGUFDequantizer();
        byte[] block = new byte[18];
        // scale = 0.5 FP16 = 0x3800 → bytes [0x00, 0x38]
        block[0] = 0x00;
        block[1] = 0x38;
        // First byte: lo nibble = 0 (→ -8*0.5 = -4.0), hi nibble = 15 (→ 7*0.5 = 3.5)
        block[2] = (byte) 0xF0; // lo=0, hi=15
        // Fill rest with 0x88 (neutral)
        for (int i = 3; i < 18; i++)
            block[i] = (byte) 0x88;

        float[] out = new float[32];
        dq.dequantQ4_0(block, 32, out);
        assertEquals(-4.0f, out[0], 0.05f, "lo nibble 0 → -4.0");
        assertEquals(3.5f, out[1], 0.05f, "hi nibble 15 → 3.5");
    }

    @Test
    @Order(3)
    @DisplayName("GGUF Q8_0: simple single block")
    void testGgufQ8_0() {
        GGUFDequantizer dq = new GGUFDequantizer();
        byte[] block = new byte[34];
        // scale = 1.0 FP16
        block[0] = 0x00;
        block[1] = 0x3C;
        // All INT8 weights = 1
        for (int i = 2; i < 34; i++)
            block[i] = (byte) 1;

        float[] out = new float[32];
        dq.dequantQ8_0(block, 32, out);
        for (float v : out)
            assertEquals(1.0f, v, 0.01f, "INT8=1, scale=1 → 1.0");
    }

    @Test
    @Order(4)
    @DisplayName("GGUF Q8_0: negative INT8 weights")
    void testGgufQ8_0Negative() {
        GGUFDequantizer dq = new GGUFDequantizer();
        byte[] block = new byte[34];
        block[0] = 0x00;
        block[1] = 0x3C; // scale=1.0
        for (int i = 2; i < 34; i++)
            block[i] = (byte) (-5); // INT8 -5

        float[] out = new float[32];
        dq.dequantQ8_0(block, 32, out);
        for (float v : out)
            assertEquals(-5.0f, v, 0.01f);
    }

    @Test
    @Order(5)
    @DisplayName("GGUF dispatch: F32 pass-through")
    void testGgufF32Passthrough() {
        GGUFDequantizer dq = new GGUFDequantizer();
        // 2 FP32 values: 1.0 and -2.5
        byte[] raw = new byte[8];
        int bits1 = Float.floatToIntBits(1.0f);
        int bits2 = Float.floatToIntBits(-2.5f);
        for (int b = 0; b < 4; b++)
            raw[b] = (byte) (bits1 >> (b * 8));
        for (int b = 0; b < 4; b++)
            raw[4 + b] = (byte) (bits2 >> (b * 8));

        float[] out = new float[2];
        dq.dequantize(GGUFDequantizer.GGMLType.F32, raw, 2, out);
        assertEquals(1.0f, out[0], 1e-5f);
        assertEquals(-2.5f, out[1], 1e-5f);
    }

    @Test
    @Order(6)
    @DisplayName("GGUF GGMLType: properties")
    void testGgmlTypeProperties() {
        assertEquals(18, GGUFDequantizer.GGMLType.Q4_0.blockBytes);
        assertEquals(34, GGUFDequantizer.GGMLType.Q8_0.blockBytes);
        assertEquals(144, GGUFDequantizer.GGMLType.Q4_K.blockBytes);
        assertEquals(210, GGUFDequantizer.GGMLType.Q6_K.blockBytes);
        assertEquals(8.0, GGUFDequantizer.GGMLType.Q4_0.compressionRatio(), 0.01);
        assertEquals(4.0, GGUFDequantizer.GGMLType.Q8_0.compressionRatio(), 0.01);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BitsAndBytes NF4 Tests
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @Order(7)
    @DisplayName("BnB NF4: table has 16 entries in [-1, +1]")
    void testNf4TableValid() {
        assertEquals(16, BnBDequantizer.NF4_TABLE.length);
        for (float v : BnBDequantizer.NF4_TABLE) {
            assertTrue(v >= -1.0f && v <= 1.0f,
                    "NF4 value out of range: " + v);
        }
        // Table must be sorted ascending
        for (int i = 1; i < BnBDequantizer.NF4_TABLE.length; i++) {
            assertTrue(BnBDequantizer.NF4_TABLE[i] > BnBDequantizer.NF4_TABLE[i - 1],
                    "NF4 table not sorted at i=" + i);
        }
    }

    @Test
    @Order(8)
    @DisplayName("BnB NF4: index 7 = 0.0 (the zero level)")
    void testNf4TableZero() {
        assertEquals(0.0f, BnBDequantizer.NF4_TABLE[7], 1e-6f,
                "NF4 index 7 should be 0.0");
    }

    @Test
    @Order(9)
    @DisplayName("BnB NF4: quantize → dequant round-trip")
    void testNf4RoundTrip() {
        BnBDequantizer dq = new BnBDequantizer(64, false);
        float[] original = { 0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 0.25f, -0.25f, 0.75f };
        // Pad to blockSize=64
        float[] weights = new float[64];
        System.arraycopy(original, 0, weights, 0, original.length);

        byte[] packed = new byte[32]; // 64/2
        float[] absmax = new float[1]; // 1 block

        dq.quantizeNF4(weights, packed, absmax);
        assertTrue(absmax[0] > 0, "absmax should be positive");

        float[] recovered = new float[64];
        dq.dequantNF4(packed, absmax, 64, recovered);

        // NF4 has finite precision — allow per-level quantization error
        float tol = 0.1f; // NF4 levels are ~0.1 apart on average
        for (int i = 0; i < original.length; i++) {
            assertEquals(original[i], recovered[i], tol,
                    "NF4 round-trip at i=" + i + ": original=" + original[i]);
        }
    }

    @Test
    @Order(10)
    @DisplayName("BnB NF4: absmax=0 → all outputs = 0")
    void testNf4AbsmaxZero() {
        BnBDequantizer dq = new BnBDequantizer(64, false);
        byte[] packed = new byte[32];
        float[] absmax = { 0.0f };
        float[] out = new float[64];
        dq.dequantNF4(packed, absmax, 64, out);
        for (float v : out)
            assertEquals(0.0f, v, 1e-6f);
    }

    @Test
    @Order(11)
    @DisplayName("BnB INT8: column-wise dequant")
    void testBnbInt8ColWise() {
        BnBDequantizer dq = new BnBDequantizer();
        // 2 rows × 4 cols, INT8 weights = [[1,2,3,4],[5,6,7,8]]
        byte[] w = { 1, 2, 3, 4, 5, 6, 7, 8 };
        // col scales = [0.5, 1.0, 2.0, 0.25]
        float[] scales = { 0.5f, 1.0f, 2.0f, 0.25f };
        float[] out = new float[8];
        dq.dequantInt8(w, scales, 2, 4, out);

        // row 0: [1*0.5, 2*1.0, 3*2.0, 4*0.25] = [0.5, 2.0, 6.0, 1.0]
        assertEquals(0.5f, out[0], 0.001f);
        assertEquals(2.0f, out[1], 0.001f);
        assertEquals(6.0f, out[2], 0.001f);
        assertEquals(1.0f, out[3], 0.001f);
        // row 1: [5*0.5, 6*1.0, 7*2.0, 8*0.25] = [2.5, 6.0, 14.0, 2.0]
        assertEquals(2.5f, out[4], 0.001f);
        assertEquals(6.0f, out[5], 0.001f);
        assertEquals(14.0f, out[6], 0.001f);
        assertEquals(2.0f, out[7], 0.001f);
    }

    @Test
    @Order(12)
    @DisplayName("BnB NF4: FP4 table has 16 entries")
    void testFp4Table() {
        // FP4 dequant: all values at zero index → 0.0
        BnBDequantizer dq = new BnBDequantizer(64, false);
        // Pack all nibbles = 0
        byte[] packed = new byte[32]; // 64 elements, all nibble=0
        float[] absmax = { 1.0f };
        float[] out = new float[64];
        dq.dequantFP4(packed, absmax, 64, out);
        // FP4_TABLE[0] = 0.0, so all outputs = 0.0 * 1.0 = 0.0
        for (float v : out)
            assertEquals(0.0f, v, 1e-6f);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HQQ Tests
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @Order(13)
    @DisplayName("HQQ: additive zero (w = scale×q + zero)")
    void testHqqAdditiveZero() {
        HQQDequantizer dq = new HQQDequantizer(4, 8, HQQDequantizer.QuantAxis.INPUT);
        int inF = 8, outF = 8;

        // pack q=0 everywhere, scale=1.0, zero=5.0 → w=5.0
        int[] qweight = new int[inF]; // all zeros → q=0 for all bit-slices
        float[] scales = new float[outF];
        java.util.Arrays.fill(scales, 1.0f);
        float[] zeros = new float[outF];
        java.util.Arrays.fill(zeros, 5.0f);
        // scales shape: [numGroups=1, outF=8] — index [g*outF+j]

        float[] out = new float[outF * inF];
        dq.dequantize(qweight, scales, zeros, inF, outF, out);

        // w = scale × 0 + zero = 5.0 for all elements
        for (float v : out)
            assertEquals(5.0f, v, 0.01f, "HQQ additive zero=5 at all positions");
    }

    @Test
    @Order(14)
    @DisplayName("HQQ: scale doubles output")
    void testHqqScaleDoubles() {
        HQQDequantizer dq = new HQQDequantizer(4, 8, HQQDequantizer.QuantAxis.INPUT);
        int inF = 8, outF = 8;

        // q=4 everywhere, zero=0, scale=2.0 → w=8.0
        int packed4 = 0;
        for (int b = 0; b < 8; b++)
            packed4 |= (4 & 0xF) << (b * 4);
        int[] qweight = new int[inF];
        java.util.Arrays.fill(qweight, packed4);

        float[] scales = new float[outF];
        java.util.Arrays.fill(scales, 2.0f);
        float[] zeros = new float[outF]; // all zero

        float[] out = new float[outF * inF];
        dq.dequantize(qweight, scales, zeros, inF, outF, out);
        for (float v : out)
            assertEquals(8.0f, v, 0.01f, "q=4, scale=2, zero=0 → 8.0");
    }

    @Test
    @Order(15)
    @DisplayName("HQQ matVec: fused dequant + dot product")
    void testHqqMatVec() {
        // w = q×0 + 1 = 1.0 everywhere, input = [1,1,...,1]
        // output[j] = sum_{i=0}^{7}(1.0 × 1.0) = 8.0
        HQQDequantizer dq = new HQQDequantizer(4, 8, HQQDequantizer.QuantAxis.INPUT);
        int inF = 8, outF = 8;

        int[] qweight = new int[inF]; // q=0
        float[] scales = new float[outF];
        java.util.Arrays.fill(scales, 0.0f);
        float[] zeros = new float[outF];
        java.util.Arrays.fill(zeros, 1.0f); // zero=1 → w=1

        float[] inputVec = new float[inF];
        java.util.Arrays.fill(inputVec, 1.0f);
        float[] outputVec = new float[outF];

        dq.dequantMatVec(qweight, scales, zeros, inputVec, outputVec, inF, outF);

        for (int j = 0; j < outF; j++) {
            assertEquals(8.0f, outputVec[j], 0.1f, "HQQ matVec at j=" + j);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SqueezeLLM Tests
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @Order(16)
    @DisplayName("SqueezeLLM: dense lookup dequant with identity codebook")
    void testSqueezeLLMDenseLookup() {
        // Identity codebook: lookup[col, k] = k (levels 0..15)
        SqueezeLLMDequantizer dq = new SqueezeLLMDequantizer(4);
        int inF = 8, outF = 8, numLevels = 16;

        // qweight: pack q=3 everywhere (all nibbles = 3)
        int packed3 = 0;
        for (int b = 0; b < 8; b++)
            packed3 |= (3 & 0xF) << (b * 4);
        int[] qweight = new int[outF / 8 * inF]; // [1, 8]
        java.util.Arrays.fill(qweight, packed3);

        // Codebook: lookup[j][k] = k (identity: dequant value = quantized index)
        float[] lookup = new float[inF * numLevels];
        for (int j = 0; j < inF; j++)
            for (int k = 0; k < numLevels; k++)
                lookup[j * numLevels + k] = (float) k;

        float[] out = new float[outF * inF];
        dq.dequantizeDense(qweight, lookup, inF, outF, out);

        // All q=3 → lookup[col, 3] = 3.0
        for (float v : out)
            assertEquals(3.0f, v, 0.001f, "identity lookup of q=3 should be 3.0");
    }

    @Test
    @Order(17)
    @DisplayName("SqueezeLLM: sparse scatter-add")
    void testSqueezeLLMSparseScatter() {
        SqueezeLLMDequantizer dq = new SqueezeLLMDequantizer(4);
        int outF = 4, inF = 4;

        float[] output = new float[outF * inF]; // all zeros
        // Sparse: row 0, col 2 += 7.5; row 2, col 1 += -3.0
        float[] nnz = { 7.5f, -3.0f };
        int[] denseIdx = { 2, 1 };
        int[] densePtr = { 0, 1, 1, 2, 2 }; // row 0: [0,1), row 1: [1,1), row 2: [1,2), row 3: [2,2)

        dq.scatterAddSparse(output, nnz, denseIdx, densePtr, outF, inF);

        assertEquals(7.5f, output[0 * inF + 2], 0.001f, "row0 col2 += 7.5");
        assertEquals(-3.0f, output[2 * inF + 1], 0.001f, "row2 col1 += -3.0");
        // All others should remain 0
        assertEquals(0.0f, output[0 * inF + 0], 0.001f);
        assertEquals(0.0f, output[1 * inF + 1], 0.001f);
    }

    @Test
    @Order(18)
    @DisplayName("SqueezeLLM: sparsity stats from CSR")
    void testSqueezeLLMSparsityStats() {
        SqueezeLLMDequantizer dq = new SqueezeLLMDequantizer(4);
        int[] densePtr = { 0, 2, 2, 5, 5 }; // 2+0+3+0 = 5 nnz, outF=4, inF=100
        SqueezeLLMDequantizer.SparsityStats stats = dq.computeSparsity(densePtr, 4, 100);
        assertEquals(5, stats.nnz());
        assertEquals(400, stats.totalWeights());
        assertEquals(1.25, stats.sparsityPct(), 0.01);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // QuantizerRegistry Tests
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @Order(19)
    @DisplayName("Registry: GGUF detected from file extension")
    void testRegistryGgufExtension() throws Exception {
        // Create a temp .gguf path (file doesn't need to exist for extension check)
        java.nio.file.Path p = java.nio.file.Path.of("/tmp/test.gguf");
        QuantizerRegistry.Detection d = QuantizerRegistry.detect(p);
        assertEquals(QuantizerRegistry.QuantFormat.GGUF, d.format());
        assertEquals(QuantizerRegistry.Detection.Confidence.HIGH, d.confidence());
        assertTrue(d.evidence().contains("extension"));
    }

    @Test
    @Order(20)
    @DisplayName("Registry: all formats have non-empty descriptions")
    void testRegistryAllFormats() {
        for (QuantizerRegistry.QuantFormat f : QuantizerRegistry.QuantFormat.values()) {
            assertNotNull(f.shortName);
            assertNotNull(f.description);
            assertFalse(f.shortName.isEmpty());
            assertFalse(f.description.isEmpty());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GGUF GGMLType.fromId Tests
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @Order(21)
    @DisplayName("GGUF: GGMLType.fromId lookup")
    void testGgmlTypeFromId() {
        assertEquals(GGUFDequantizer.GGMLType.Q4_0, GGUFDequantizer.GGMLType.fromId(2));
        assertEquals(GGUFDequantizer.GGMLType.Q8_0, GGUFDequantizer.GGMLType.fromId(8));
        assertEquals(GGUFDequantizer.GGMLType.Q4_K, GGUFDequantizer.GGMLType.fromId(12));
        assertEquals(GGUFDequantizer.GGMLType.F32, GGUFDequantizer.GGMLType.fromId(999));
    }

    @Test
    @Order(22)
    @DisplayName("GGUF: all block sizes are positive")
    void testGgmlBlockSizes() {
        for (GGUFDequantizer.GGMLType t : GGUFDequantizer.GGMLType.values()) {
            assertTrue(t.blockBytes > 0, t.name() + " blockBytes must be positive");
            assertTrue(t.blockSize > 0, t.name() + " blockSize must be positive");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HQQ Axis=OUTPUT Tests
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @Order(23)
    @DisplayName("HQQ: axis=OUTPUT uses output-dimension groups")
    void testHqqAxisOutput() {
        // axis=OUTPUT: scale/zero shaped [numGroups_outF, inF]
        HQQDequantizer dq = new HQQDequantizer(4, 8, HQQDequantizer.QuantAxis.OUTPUT);
        int inF = 8, outF = 8;

        int[] qweight = new int[inF]; // q=0
        // scale [numGroups=1, inF=8] = 0 everywhere → w = 0×0 + zero
        float[] scales = new float[inF]; // 1 group × 8 inF = 8 entries
        float[] zeros = new float[inF];
        java.util.Arrays.fill(zeros, 3.0f);

        float[] out = new float[outF * inF];
        dq.dequantize(qweight, scales, zeros, inF, outF, out);
        for (float v : out)
            assertEquals(3.0f, v, 0.01f, "axis=OUTPUT: zero=3 everywhere");
    }

    @Test
    @Order(24)
    @DisplayName("SqueezeLLM: full dequant (dense+sparse) on synthetic data")
    void testSqueezeLLMFullDequant() {
        SqueezeLLMDequantizer dq = new SqueezeLLMDequantizer(4);
        int inF = 8, outF = 8, numLevels = 16;

        // All q=0, identity lookup → dense = 0
        int[] qweight = new int[outF / 8 * inF];
        float[] lookup = new float[inF * numLevels]; // all zeros
        for (int j = 0; j < inF; j++)
            for (int k = 0; k < numLevels; k++)
                lookup[j * numLevels + k] = (float) k;

        // Sparse: add 5.0 at row=0, col=0
        float[] nnz = { 5.0f };
        int[] denseIdx = { 0 };
        int[] densePtr = new int[outF + 1]; // all zeros except last entry = 1
        densePtr[1] = 1; // row 0 has 1 nnz

        float[] out = new float[outF * inF];
        dq.dequantize(qweight, lookup, nnz, denseIdx, densePtr, inF, outF, out);

        assertEquals(5.0f, out[0], 0.001f, "sparse at [0,0] += 5.0");
        assertEquals(0.0f, out[1], 0.001f, "rest should be 0");
        assertEquals(0.0f, out[inF + 0], 0.001f, "row 1 should be 0");
    }
}
