package tech.kayys.gollek.gguf.loader;

import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GGUF/GGML Dequantization Engine — JDK 25 Vector API.
 *
 * GGUF (GPT-Generated Unified Format) uses block-based quantization where
 * weights are organized into fixed-size blocks, each with its own scale.
 */
public class GGUFDequantizer {

    private static final Logger log = LoggerFactory.getLogger(GGUFDequantizer.class);

    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();

    /** GGUF quantization type identifiers (from ggml.h) */
    public enum GGMLType {
        Q4_0(2, 18, 32, 4, false),
        Q4_1(3, 20, 32, 4, true),
        Q5_0(6, 22, 32, 5, false),
        Q5_1(7, 24, 32, 5, true),
        Q8_0(8, 34, 32, 8, false),
        Q2_K(10, 84, 256, 2, true),
        Q3_K(11, 110, 256, 3, false),
        Q4_K(12, 144, 256, 4, true),
        Q5_K(13, 176, 256, 5, true),
        Q6_K(14, 210, 256, 6, false),
        F16(1, 2, 1, 16, false),
        F32(0, 4, 1, 32, false);

        public final int ggmlId;
        public final int blockBytes;
        public final int blockSize; 
        public final int bits;
        public final boolean hasMin; 

        GGMLType(int id, int blockBytes, int blockSize, int bits, boolean hasMin) {
            this.ggmlId = id;
            this.blockBytes = blockBytes;
            this.blockSize = blockSize;
            this.bits = bits;
            this.hasMin = hasMin;
        }

        public double compressionRatio() {
            return 32.0 / bits;
        }

        public static GGMLType fromId(int id) {
            for (GGMLType t : values())
                if (t.ggmlId == id)
                    return t;
            return F32;
        }
    }

    public void dequantQ4_0(byte[] rawBlocks, int numElements, float[] output) {
        final int BLOCK_SIZE = 32;
        final int BLOCK_BYTES = 18;
        int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < numBlocks; b++) {
            int blockOff = b * BLOCK_BYTES;
            int outBase = b * BLOCK_SIZE;

            short dBits = (short) (((rawBlocks[blockOff + 1] & 0xFF) << 8)
                    | (rawBlocks[blockOff] & 0xFF));
            float d = fp16ToFloat(dBits);

            int nib = 0;
            while (nib < BLOCK_SIZE) {
                int j = nib / 2;
                int packed = rawBlocks[blockOff + 2 + j] & 0xFF;
                int lo = (packed & 0x0F) - 8;
                int hi = (packed >> 4) - 8;

                int outIdx = outBase + nib;
                if (outIdx < numElements)
                    output[outIdx] = d * lo;
                if (outIdx + 1 < numElements)
                    output[outIdx + 1] = d * hi;
                nib += 2;
            }
        }
    }

    public void dequantQ8_0(byte[] rawBlocks, int numElements, float[] output) {
        final int BLOCK_SIZE = 32;
        final int BLOCK_BYTES = 34;
        int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < numBlocks; b++) {
            int blockOff = b * BLOCK_BYTES;
            int outBase = b * BLOCK_SIZE;

            short dBits = (short) (((rawBlocks[blockOff + 1] & 0xFF) << 8)
                    | (rawBlocks[blockOff] & 0xFF));
            float d = fp16ToFloat(dBits);

            int i = 0;
            for (; i <= BLOCK_SIZE - F_LANES && outBase + i + F_LANES <= numElements; i += F_LANES) {
                float[] vals = new float[F_LANES];
                for (int lane = 0; lane < F_LANES; lane++) {
                    vals[lane] = (float) rawBlocks[blockOff + 2 + i + lane];
                }
                FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, vals, 0);
                FloatVector vd = FloatVector.broadcast(FLOAT_SPECIES, d);
                vq.mul(vd).intoArray(output, outBase + i);
            }
            for (; i < BLOCK_SIZE && outBase + i < numElements; i++) {
                output[outBase + i] = d * rawBlocks[blockOff + 2 + i];
            }
        }
    }

    public void dequantQ4K(byte[] rawBlocks, int numElements, float[] output) {
        final int SUPER_BLOCK_SIZE = 256;
        final int SUPER_BLOCK_BYTES = 144;
        final int NUM_SUB = 8; 
        final int SUB_SIZE = 32;

        int numSuperBlocks = (numElements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;

        for (int sb = 0; sb < numSuperBlocks; sb++) {
            int sbOff = sb * SUPER_BLOCK_BYTES;
            int outBase = sb * SUPER_BLOCK_SIZE;

            float d = fp16ToFloat(readFp16(rawBlocks, sbOff));
            float dmin = fp16ToFloat(readFp16(rawBlocks, sbOff + 2));

            float[] sc = new float[NUM_SUB];
            float[] mn = new float[NUM_SUB];
            decodeQ4KScales(rawBlocks, sbOff + 4, d, dmin, sc, mn);

            int qsOff = sbOff + 16; 

            for (int sub = 0; sub < NUM_SUB; sub++) {
                int subOutBase = outBase + sub * SUB_SIZE;
                float scale = sc[sub];
                float minV = mn[sub];

                int halfOff = qsOff + sub * 16; 
                int hiOff = qsOff + 64 + sub * 16; 

                for (int i = 0; i < 16 && subOutBase + i < numElements; i++) {
                    int bLo = rawBlocks[halfOff + i] & 0xFF;
                    int bHi = rawBlocks[hiOff + i] & 0xFF;
                    int lo = bLo & 0xF;
                    int hi = bHi & 0xF;

                    output[subOutBase + i] = scale * lo - minV;
                    if (subOutBase + i + 16 < numElements)
                        output[subOutBase + i + 16] = scale * hi - minV;
                }
            }
        }
    }

    private void decodeQ4KScales(byte[] data, int off,
            float d, float dmin,
            float[] sc, float[] mn) {
        for (int i = 0; i < 8; i++) {
            int lo = data[off + i] & 0xFF;
            int scLo = lo & 0x3F; 
            int mnLo = (data[off + i + 4] >> 0) & 0x3F;
            sc[i] = d * scLo;
            mn[i] = dmin * mnLo;
        }
    }

    public void dequantQ5K(byte[] rawBlocks, int numElements, float[] output) {
        final int SUPER_BLOCK_BYTES = 176;
        final int SUPER_BLOCK_SIZE = 256;
        final int NUM_SUB = 8;

        int numSuperBlocks = (numElements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;

        for (int sb = 0; sb < numSuperBlocks; sb++) {
            int sbOff = sb * SUPER_BLOCK_BYTES;
            int outBase = sb * SUPER_BLOCK_SIZE;

            float d = fp16ToFloat(readFp16(rawBlocks, sbOff));
            float dmin = fp16ToFloat(readFp16(rawBlocks, sbOff + 2));

            float[] sc = new float[NUM_SUB];
            float[] mn = new float[NUM_SUB];
            decodeQ4KScales(rawBlocks, sbOff + 4, d, dmin, sc, mn);

            int hiBitsOff = sbOff + 16; 
            int qsOff = sbOff + 48; 

            for (int sub = 0; sub < NUM_SUB; sub++) {
                int subOut = outBase + sub * 32;
                float scale = sc[sub], minV = mn[sub];

                for (int i = 0; i < 32 && subOut + i < numElements; i++) {
                    int byteIdx = sub * 4 + i / 8;
                    int bit = (rawBlocks[hiBitsOff + byteIdx] >> (i % 8)) & 1;
                    int nibIdx = sub * 16 + i / 2;
                    int nibble = (rawBlocks[qsOff + nibIdx] >> ((i % 2) * 4)) & 0xF;
                    int q5 = (bit << 4) | nibble;
                    output[subOut + i] = scale * q5 - minV;
                }
            }
        }
    }

    public void dequantQ6K(byte[] rawBlocks, int numElements, float[] output) {
        final int SUPER_BLOCK_BYTES = 210;
        final int SUPER_BLOCK_SIZE = 256;
        final int NUM_SUB = 16;

        int numSuperBlocks = (numElements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;

        for (int sb = 0; sb < numSuperBlocks; sb++) {
            int sbOff = sb * SUPER_BLOCK_BYTES;
            int outBase = sb * SUPER_BLOCK_SIZE;

            int ql = sbOff; 
            int qh = sbOff + 128; 
            int sc = sbOff + 192; 
            int dOff = sbOff + 208; 

            float d = fp16ToFloat(readFp16(rawBlocks, dOff));

            for (int i = 0; i < SUPER_BLOCK_SIZE && outBase + i < numElements; i++) {
                int sub = i / 16;
                float subScale = d * rawBlocks[sc + sub]; 

                int qIdx = i / 2;
                int lo4 = (rawBlocks[ql + qIdx] >> ((i % 2) * 4)) & 0xF;
                int hiIdx = i / 4;
                int hi2 = (rawBlocks[qh + hiIdx] >> ((i % 4) * 2)) & 0x3;
                int q6 = lo4 | (hi2 << 4);
                output[outBase + i] = subScale * (q6 - 32); 
            }
        }
    }

    public void dequantize(GGMLType type, byte[] rawBlocks, int numElements, float[] output) {
        switch (type) {
            case Q4_0 -> dequantQ4_0(rawBlocks, numElements, output);
            case Q4_1 -> dequantQ4_1(rawBlocks, numElements, output);
            case Q5_0 -> dequantQ5_0(rawBlocks, numElements, output);
            case Q8_0 -> dequantQ8_0(rawBlocks, numElements, output);
            case Q4_K -> dequantQ4K(rawBlocks, numElements, output);
            case Q5_K -> dequantQ5K(rawBlocks, numElements, output);
            case Q6_K -> dequantQ6K(rawBlocks, numElements, output);
            case F16 -> dequantF16(rawBlocks, numElements, output);
            case F32 -> dequantF32(rawBlocks, numElements, output);
            default -> throw new UnsupportedOperationException(
                    "GGML type not yet implemented: " + type);
        }
        log.debug("Dequantized {} elements using {}", numElements, type);
    }

    private void dequantQ4_1(byte[] blocks, int numElements, float[] output) {
        final int BLOCK_SIZE = 32, BLOCK_BYTES = 20;
        int n = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int b = 0; b < n; b++) {
            int bOff = b * BLOCK_BYTES, outBase = b * BLOCK_SIZE;
            float d = fp16ToFloat(readFp16(blocks, bOff));
            float m = fp16ToFloat(readFp16(blocks, bOff + 2));
            for (int i = 0; i < BLOCK_SIZE && outBase + i < numElements; i++) {
                int packed = blocks[bOff + 4 + i / 2] & 0xFF;
                int q = (i % 2 == 0) ? (packed & 0xF) : (packed >> 4);
                output[outBase + i] = d * q + m;
            }
        }
    }

    private void dequantQ5_0(byte[] blocks, int numElements, float[] output) {
        final int BLOCK_SIZE = 32, BLOCK_BYTES = 22;
        int n = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int b = 0; b < n; b++) {
            int bOff = b * BLOCK_BYTES, outBase = b * BLOCK_SIZE;
            float d = fp16ToFloat(readFp16(blocks, bOff));
            int qhInt = ((blocks[bOff + 2] & 0xFF)) | ((blocks[bOff + 3] & 0xFF) << 8)
                    | ((blocks[bOff + 4] & 0xFF) << 16) | ((blocks[bOff + 5] & 0xFF) << 24);
            for (int i = 0; i < BLOCK_SIZE && outBase + i < numElements; i++) {
                int nibIdx = bOff + 6 + i / 2;
                int lo4 = (i % 2 == 0) ? (blocks[nibIdx] & 0xF) : ((blocks[nibIdx] >> 4) & 0xF);
                int hi1 = (qhInt >> i) & 1;
                int q5 = lo4 | (hi1 << 4);
                output[outBase + i] = d * (q5 - 16);
            }
        }
    }

    private void dequantF16(byte[] blocks, int numElements, float[] output) {
        for (int i = 0; i < numElements; i++) {
            output[i] = fp16ToFloat(readFp16(blocks, i * 2));
        }
    }

    private void dequantF32(byte[] blocks, int numElements, float[] output) {
        for (int i = 0; i < numElements; i++) {
            int bits = ((blocks[i * 4] & 0xFF)) | ((blocks[i * 4 + 1] & 0xFF) << 8)
                    | ((blocks[i * 4 + 2] & 0xFF) << 16) | ((blocks[i * 4 + 3] & 0xFF) << 24);
            output[i] = Float.intBitsToFloat(bits);
        }
    }

    private static short readFp16(byte[] data, int off) {
        return (short) (((data[off + 1] & 0xFF) << 8) | (data[off] & 0xFF));
    }

    private static float fp16ToFloat(short fp16) {
        int h = fp16 & 0xFFFF;
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0)
            return Float.intBitsToFloat((sign << 31) | (mant == 0 ? 0 : ((mant << 13) | ((127 - 15 - 1) << 23))));
        if (exp == 31)
            return Float.intBitsToFloat((sign << 31) | 0x7F800000 | (mant << 13));
        return Float.intBitsToFloat((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13));
    }

    /**
     * Fast F16 dequantization using batched reads.
     * Reads chunks of F16 values at once via MemorySegment.copy to avoid
     * per-element bounds-checked reads.
     */
    public static void dequantizeF16(java.lang.foreign.MemorySegment raw, java.lang.foreign.MemorySegment f32, long numElements) {
        dequantizeF16(raw, 0, f32, numElements);
    }

    public static void dequantizeF16(java.lang.foreign.MemorySegment raw, long rawOffset, java.lang.foreign.MemorySegment f32, long numElements) {
        for (long i = 0; i < numElements; i++) {
            short fp16 = raw.get(java.lang.foreign.ValueLayout.JAVA_SHORT, rawOffset + i * 2L);
            float val = fp16ToFloatDirect(fp16);
            f32.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, i * Float.BYTES, val);
        }
    }

    /**
     * Fast Q8_0 dequantization using batched block reads.
     * Reads entire 34-byte blocks at once via MemorySegment.copy to avoid
     * per-byte bounds-checked reads (which are ~20x slower).
     *
     * Q8_0 block layout: 2 bytes FP16 scale + 32 signed int8 quants = 34 bytes/block.
     */
    public static void dequantizeQ8_0(java.lang.foreign.MemorySegment raw, java.lang.foreign.MemorySegment f32, long numElements) {
        dequantizeQ8_0(raw, 0, f32, numElements);
    }

    private static final VectorSpecies<Byte> B8_SPECIES = VectorSpecies.of(byte.class, VectorShape.S_64_BIT);
    private static final VectorSpecies<Float> F8_SPECIES = VectorSpecies.of(float.class, VectorShape.S_256_BIT);

    public static void dequantizeQ8_0(java.lang.foreign.MemorySegment raw, long rawOffset, java.lang.foreign.MemorySegment f32, long numElements) {
        final int BLOCK_SIZE = 32;
        final int BLOCK_BYTES = 34;
        long numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (long b = 0; b < numBlocks; b++) {
            long blockOff = rawOffset + b * BLOCK_BYTES;
            long outBase = b * BLOCK_SIZE;
            int elemsInBlock = (int) Math.min(BLOCK_SIZE, numElements - outBase);

            // Read FP16 scale directly from mmap'd segment
            short dBits = raw.get(java.lang.foreign.ValueLayout.JAVA_SHORT, blockOff);
            float d = fp16ToFloatDirect(dBits);

            // Dequantize int8 quants using Vector API
            int i = 0;
            int limit = elemsInBlock - (elemsInBlock % 8);
            for (; i < limit; i += 8) {
                ByteVector qBytes = ByteVector.fromMemorySegment(B8_SPECIES, raw, blockOff + 2 + i, java.nio.ByteOrder.nativeOrder());
                FloatVector qFloats = (FloatVector) qBytes.castShape(F8_SPECIES, 0);
                FloatVector result = qFloats.mul(d);
                result.intoMemorySegment(f32, (outBase + i) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            }
            
            // Tail loop for leftovers
            for (; i < elemsInBlock; i++) {
                byte q = raw.get(java.lang.foreign.ValueLayout.JAVA_BYTE, blockOff + 2 + i);
                f32.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, (outBase + i) * Float.BYTES, d * q);
            }
        }
    }

    /** FP16 to float conversion for direct MemorySegment path. */
    private static float fp16ToFloatDirect(short fp16) {
        int h = fp16 & 0xFFFF;
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0)
            return Float.intBitsToFloat((sign << 31) | (mant == 0 ? 0 : ((mant << 13) | ((127 - 15 - 1) << 23))));
        if (exp == 31)
            return Float.intBitsToFloat((sign << 31) | 0x7F800000 | (mant << 13));
        return Float.intBitsToFloat((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13));
    }

    public static void printCapabilities() {
        System.out.println("=== GGUF Dequantizer ===");
        System.out.printf("Supported types: Q4_0 Q4_1 Q5_0 Q8_0 Q4_K Q5_K Q6_K F16 F32%n");
        System.out.printf("SIMD: %s (%d float lanes)%n", FLOAT_SPECIES.toString(), F_LANES);
    }
}
