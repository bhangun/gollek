package tech.kayys.gollek.quantizer.gptq;

import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * GPTQ Dequantization Engine using JDK 25 Vector API.
 *
 * The Vector API enables explicit SIMD (Single Instruction Multiple Data)
 * operations using CPU vector units (SSE4, AVX2, AVX-512 on x86;
 * NEON/SVE on ARM). The JIT compiler maps Vector API operations directly
 * to hardware intrinsics.
 *
 * GPTQ Dequantization Formula (INT4 asymmetric, per group g):
 * ─────────────────────────────────────────────────────────────
 * w_dequant[i, j] = (q[i, j] - zeros[g, j]) * scales[g, j]
 *
 * where:
 * q = unpacked 4-bit quantized weight (range [0, 15])
 * zeros = unpacked 4-bit zero-point (range [0, 15])
 * scales = FP16 scale per (group, output_feature)
 * g = i / group_size (which group row i belongs to)
 *
 * The INT32 packing stores multiple quantized values per INT32 word:
 * For 4-bit: 8 values per INT32 (bits 0-3, 4-7, 8-11, ..., 28-31)
 * For 8-bit: 4 values per INT32 (bits 0-7, 8-15, 16-23, 24-31)
 * For 3-bit: 10 values per INT32 with overlap handling
 *
 * Vector API strategy:
 * - Use IntVector to load and unpack 8× INT4 values in parallel
 * - Use FloatVector for scale multiplication and accumulation
 * - Process FLOAT_SPECIES.length() elements per SIMD iteration
 */
public class VectorDequantizer {

    private static final Logger log = LoggerFactory.getLogger(VectorDequantizer.class);

    // ── Vector Species ────────────────────────────────────────────────────────
    // Preferred species = widest hardware vector; JIT selects AVX-512 if available

    /** 32-bit integer SIMD species (used for unpacking quantized weights) */
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;

    /** 32-bit float SIMD species (used for dequantized output) */
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    /** Number of float lanes in the preferred SIMD width */
    private static final int F_LANES = FLOAT_SPECIES.length();

    /** Number of int lanes in the preferred SIMD width */
    private static final int I_LANES = INT_SPECIES.length();

    private final GPTQConfig config;
    private final int packFactor; // elements per INT32 (8 for 4-bit)
    private final int quantMask; // bitmask per element (0xF for 4-bit)

    public VectorDequantizer(GPTQConfig config) {
        this.config = config;
        this.packFactor = config.elementsPerInt32();
        this.quantMask = config.quantMask();

        log.info("VectorDequantizer: bits={}, pack={}, mask=0x{}, SIMD width={} floats ({})",
                config.bits(), packFactor,
                Integer.toHexString(quantMask),
                F_LANES, FLOAT_SPECIES.toString());
    }

    /**
     * Returns the bit-width this dequantizer is configured for.
     *
     * @return bit-width (e.g. 4)
     */
    public int getBits() {
        return config.bits();
    }

    // ── Main Dequantization Entry Point ───────────────────────────────────────

    /**
     * Dequantizes a full GPTQ layer into a flat FP32 output array.
     *
     * @param qweightInts  packed INT32 quantized weights, shape [rows, cols] where
     *                     rows = outFeatures/packFactor, cols = inFeatures
     * @param qzerosInts   packed INT32 zero-points, shape [numGroups,
     *                     outFeatures/packFactor]
     * @param scalesShorts FP16 scales (as raw shorts), shape [numGroups,
     *                     outFeatures]
     * @param gIdxInts     optional group index per input feature (null if no
     *                     act-order)
     * @param inFeatures   K dimension
     * @param outFeatures  N dimension
     * @param output       pre-allocated FP32 output buffer, size [outFeatures *
     *                     inFeatures]
     */
    public void dequantize(
            int[] qweightInts,
            int[] qzerosInts,
            short[] scalesShorts,
            int[] gIdxInts, // nullable
            int inFeatures,
            int outFeatures,
            float[] output) {
        log.debug("Dequantizing: inFeatures={}, outFeatures={}, bits={}, groupSize={}",
                inFeatures, outFeatures, config.bits(), config.groupSize());

        if (config.bits() == 4) {
            dequantize4Bit(qweightInts, qzerosInts, scalesShorts, gIdxInts,
                    inFeatures, outFeatures, output);
        } else if (config.bits() == 8) {
            dequantize8Bit(qweightInts, qzerosInts, scalesShorts, gIdxInts,
                    inFeatures, outFeatures, output);
        } else {
            // Generic path for 2-bit and 3-bit
            dequantizeGeneric(qweightInts, qzerosInts, scalesShorts, gIdxInts,
                    inFeatures, outFeatures, output);
        }
    }

    // ── 4-bit Specialized Dequantization (most common) ───────────────────────

    /**
     * Optimized 4-bit dequantization with Vector API inner loop.
     *
     * Memory layout for 4-bit INT4 packed into INT32:
     * qweight[packed_row][col] packs 8 output-feature weights per element
     * Bit layout: [f7 f6 f5 f4 f3 f2 f1 f0] each 4 bits, LSB first
     *
     * The inner loop processes FLOAT_SPECIES.length() output features at once
     * using SIMD float multiply-add for scale/zero-point adjustment.
     */
    private void dequantize4Bit(
            int[] qweightInts, int[] qzerosInts, short[] scalesShorts, int[] gIdxInts,
            int inFeatures, int outFeatures, float[] output) {
        int numGroups = (inFeatures + config.groupSize() - 1) / config.groupSize();
        int packedRows = outFeatures / packFactor; // INT32 rows for qweight

        // Unpack scales from FP16 to FP32 (done once)
        float[] scales = fp16ArrayToFloat32(scalesShorts, numGroups, outFeatures);

        // Unpack zeros from INT32 packed format to FP32
        float[] zeros = unpackZerosToFloat(qzerosInts, numGroups, outFeatures);

        // Main dequantization: iterate over input features (K dimension)
        for (int k = 0; k < inFeatures; k++) {
            // Which group does this input feature belong to?
            int g = (gIdxInts != null) ? gIdxInts[k] : (k / config.groupSize());

            // Pointer into the packed qweight for column k
            // qweight layout: [outFeatures/packFactor][inFeatures]
            // For output feature j, packed word index = j / packFactor
            // Bit offset within INT32 = (j % packFactor) * bits

            // Process outFeatures in SIMD chunks of F_LANES
            int j = 0;
            for (; j <= outFeatures - F_LANES; j += F_LANES) {
                // ── SIMD dequantization for F_LANES output features ────────────

                // Gather and unpack quantized weight values
                float[] wVals = new float[F_LANES];
                float[] sVals = new float[F_LANES];
                float[] zVals = new float[F_LANES];

                for (int lane = 0; lane < F_LANES; lane++) {
                    int jj = j + lane;
                    int packedRow = jj / packFactor;
                    int bitShift = (jj % packFactor) * config.bits();
                    int qIdx = packedRow * inFeatures + k;
                    int qVal = (qweightInts[qIdx] >> bitShift) & quantMask;
                    wVals[lane] = qVal;
                    sVals[lane] = scales[g * outFeatures + jj];
                    zVals[lane] = zeros[g * outFeatures + jj];
                }

                // Load gathered values into SIMD vectors
                FloatVector vw = FloatVector.fromArray(FLOAT_SPECIES, wVals, 0);
                FloatVector vs = FloatVector.fromArray(FLOAT_SPECIES, sVals, 0);
                FloatVector vz = FloatVector.fromArray(FLOAT_SPECIES, zVals, 0);

                // GPTQ formula: result = (w - zero) * scale
                // FMA: vs.fma(-vz, vw.mul(vs)) → (vw - vz) * vs
                FloatVector result = vw.sub(vz).mul(vs);

                // Store to output[k * outFeatures + j]
                result.intoArray(output, k * outFeatures + j);
            }

            // Scalar tail (remaining output features not covered by SIMD)
            for (; j < outFeatures; j++) {
                int packedRow = j / packFactor;
                int bitShift = (j % packFactor) * config.bits();
                int qIdx = packedRow * inFeatures + k;
                int qVal = (qweightInts[qIdx] >> bitShift) & quantMask;
                float scale = scales[g * outFeatures + j];
                float zero = zeros[g * outFeatures + j];
                output[k * outFeatures + j] = (qVal - zero) * scale;
            }
        }
    }

    // ── 8-bit Specialized Dequantization ─────────────────────────────────────

    private void dequantize8Bit(
            int[] qweightInts, int[] qzerosInts, short[] scalesShorts, int[] gIdxInts,
            int inFeatures, int outFeatures, float[] output) {
        int numGroups = (inFeatures + config.groupSize() - 1) / config.groupSize();
        float[] scales = fp16ArrayToFloat32(scalesShorts, numGroups, outFeatures);
        float[] zeros = unpackZerosToFloat(qzerosInts, numGroups, outFeatures);

        for (int k = 0; k < inFeatures; k++) {
            int g = (gIdxInts != null) ? gIdxInts[k] : (k / config.groupSize());

            int j = 0;
            for (; j <= outFeatures - F_LANES; j += F_LANES) {
                float[] wVals = new float[F_LANES];
                float[] sVals = new float[F_LANES];
                float[] zVals = new float[F_LANES];

                for (int lane = 0; lane < F_LANES; lane++) {
                    int jj = j + lane;
                    int byteIdx = jj / 4;
                    int byteOff = (jj % 4) * 8;
                    int qVal = (qweightInts[byteIdx * inFeatures + k] >> byteOff) & 0xFF;
                    wVals[lane] = qVal;
                    sVals[lane] = scales[g * outFeatures + jj];
                    zVals[lane] = zeros[g * outFeatures + jj];
                }

                FloatVector vw = FloatVector.fromArray(FLOAT_SPECIES, wVals, 0);
                FloatVector vs = FloatVector.fromArray(FLOAT_SPECIES, sVals, 0);
                FloatVector vz = FloatVector.fromArray(FLOAT_SPECIES, zVals, 0);
                vw.sub(vz).mul(vs).intoArray(output, k * outFeatures + j);
            }

            // Scalar tail
            for (; j < outFeatures; j++) {
                int byteIdx = j / 4;
                int byteOff = (j % 4) * 8;
                int qVal = (qweightInts[byteIdx * inFeatures + k] >> byteOff) & 0xFF;
                output[k * outFeatures + j] = (qVal - zeros[g * outFeatures + j])
                        * scales[g * outFeatures + j];
            }
        }
    }

    // ── Generic Dequantization (2-bit, 3-bit) ────────────────────────────────

    private void dequantizeGeneric(
            int[] qweightInts, int[] qzerosInts, short[] scalesShorts, int[] gIdxInts,
            int inFeatures, int outFeatures, float[] output) {
        int numGroups = (inFeatures + config.groupSize() - 1) / config.groupSize();
        float[] scales = fp16ArrayToFloat32(scalesShorts, numGroups, outFeatures);
        float[] zeros = unpackZerosToFloat(qzerosInts, numGroups, outFeatures);

        int bits = config.bits();
        int mask = quantMask;

        for (int k = 0; k < inFeatures; k++) {
            int g = (gIdxInts != null) ? gIdxInts[k] : (k / config.groupSize());
            for (int j = 0; j < outFeatures; j++) {
                // For non-power-of-2 bit widths (e.g., 3-bit), elements can
                // span INT32 word boundaries. Compute bit position carefully.
                long bitPos = (long) j * bits;
                int wordIdx = (int) (bitPos / 32);
                int bitOff = (int) (bitPos % 32);

                int qVal;
                if (bitOff + bits <= 32) {
                    qVal = (qweightInts[wordIdx * inFeatures + k] >> bitOff) & mask;
                } else {
                    // Spans two INT32 words
                    int lo = qweightInts[wordIdx * inFeatures + k] >>> bitOff;
                    int hi = qweightInts[(wordIdx + 1) * inFeatures + k] << (32 - bitOff);
                    qVal = (lo | hi) & mask;
                }

                output[k * outFeatures + j] = (qVal - zeros[g * outFeatures + j])
                        * scales[g * outFeatures + j];
            }
        }
    }

    // ── Vector-Accelerated Matrix Multiply (W × X for inference) ─────────────

    /**
     * Computes output = weight_fp32 × input_fp32 using Vector API.
     * Used after dequantization for actual inference.
     *
     * @param weight dequantized FP32 weight [outF × inF]
     * @param input  FP32 input vector [inF]
     * @param output FP32 result vector [outF]
     * @param outF   output features (rows)
     * @param inF    input features (cols)
     */
    public void matVecMul(float[] weight, float[] input, float[] output, int outF, int inF) {
        for (int row = 0; row < outF; row++) {
            int rowOff = row * inF;
            FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
            int k = 0;

            // SIMD dot product
            for (; k <= inF - F_LANES; k += F_LANES) {
                FloatVector vw = FloatVector.fromArray(FLOAT_SPECIES, weight, rowOff + k);
                FloatVector vi = FloatVector.fromArray(FLOAT_SPECIES, input, k);
                acc = vw.fma(vi, acc); // acc += vw * vi
            }

            // Horizontal sum of SIMD accumulator
            float sum = acc.reduceLanes(VectorOperators.ADD);

            // Scalar tail
            for (; k < inF; k++) {
                sum += weight[rowOff + k] * input[k];
            }

            output[row] = sum;
        }
    }

    // ── Utility: FP16 → FP32 Array Conversion ────────────────────────────────

    /**
     * Converts FP16 scale array to FP32.
     * Vectorized using Vector API over raw bit patterns.
     */
    public float[] fp16ArrayToFloat32(short[] fp16, int rows, int cols) {
        float[] result = new float[rows * cols];
        for (int i = 0; i < result.length; i++) {
            result[i] = MemoryAllocator.fp16ToFloat32(fp16[i]);
        }
        return result;
    }

    /**
     * Unpacks INT32-packed zero-points to a FP32 zero array.
     * Same packing format as qweight.
     *
     * qzeros layout: [numGroups, outFeatures/packFactor]
     * Output layout: [numGroups, outFeatures]
     */
    public float[] unpackZerosToFloat(int[] qzerosInts, int numGroups, int outFeatures) {
        float[] result = new float[numGroups * outFeatures];
        int packedCols = outFeatures / packFactor;

        for (int g = 0; g < numGroups; g++) {
            for (int jp = 0; jp < packedCols; jp++) {
                int packed = qzerosInts[g * packedCols + jp];
                for (int b = 0; b < packFactor; b++) {
                    int j = jp * packFactor + b;
                    if (j < outFeatures) {
                        // +1 offset is standard AutoGPTQ convention
                        result[g * outFeatures + j] = (float) (((packed >> (b * config.bits())) & quantMask) + 1);
                    }
                }
            }
        }

        return result;
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    public static void printVectorCapabilities() {
        System.out.println("=== Vector API Capabilities ===");
        System.out.printf("Float species: %s (%d lanes, %d bits)%n",
                FLOAT_SPECIES.toString(), FLOAT_SPECIES.length(), FLOAT_SPECIES.vectorBitSize());
        System.out.printf("Int species:   %s (%d lanes, %d bits)%n",
                INT_SPECIES.toString(), INT_SPECIES.length(), INT_SPECIES.vectorBitSize());
        System.out.printf("Hardware SIMD: %s%n",
                FLOAT_SPECIES.vectorBitSize() >= 256 ? "AVX2/NEON+"
                        : FLOAT_SPECIES.vectorBitSize() >= 128 ? "SSE4/NEON" : "Scalar fallback");
    }
}
