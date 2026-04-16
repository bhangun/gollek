package tech.kayys.gollek.converter.gguf;

/**
 * ggml_type enum – tensor element types supported in GGUF.
 * Each entry records: on-disk id, human label, block size,
 * and the number of bytes per block (type_size in ggml terms).
 */
public enum GgmlType {
    F32(0, "F32", 1, 4),
    F16(1, "F16", 1, 2),
    Q4_0(2, "Q4_0", 32, 18), // 32 elem/block, 2 bytes header + 16 bytes weights
    Q4_1(3, "Q4_1", 32, 20),
    // 4,5 reserved
    Q5_0(6, "Q5_0", 32, 22),
    Q5_1(7, "Q5_1", 32, 24),
    Q8_0(8, "Q8_0", 32, 34), // 32 elem, 2-byte scale + 32 bytes
    Q8_1(9, "Q8_1", 32, 40),
    // K-quants
    Q2_K(10, "Q2_K", 256, 84),
    Q3_K(11, "Q3_K", 256, 110),
    Q4_K(12, "Q4_K", 256, 144),
    Q5_K(13, "Q5_K", 256, 176),
    Q6_K(14, "Q6_K", 256, 210),
    Q8_K(15, "Q8_K", 256, 292),
    // iQuants
    IQ2_XXS(16, "IQ2_XXS", 256, 66),
    IQ2_XS(17, "IQ2_XS", 256, 74),
    IQ3_XXS(18, "IQ3_XXS", 256, 98),
    IQ1_S(19, "IQ1_S", 256, 50),
    IQ4_NL(20, "IQ4_NL", 32, 18),
    IQ3_S(21, "IQ3_S", 256, 110),
    IQ2_S(22, "IQ2_S", 256, 82),
    IQ4_XS(23, "IQ4_XS", 256, 136),
    I8(24, "I8", 1, 1),
    I16(25, "I16", 1, 2),
    I32(26, "I32", 1, 4),
    I64(27, "I64", 1, 8),
    F64(28, "F64", 1, 8),
    IQ1_M(29, "IQ1_M", 256, 56),
    BF16(30, "BF16", 1, 2),
    Q4_0_4_4(31, "Q4_0_4_4", 32, 18),
    Q4_0_4_8(32, "Q4_0_4_8", 32, 18),
    Q4_0_8_8(33, "Q4_0_8_8", 32, 18);

    public final int id;
    public final String label;
    /** Number of elements per quantization block. */
    public final int blockSize;
    /** Bytes consumed per block on disk. */
    public final int typeSize;

    GgmlType(int id, String label, int blockSize, int typeSize) {
        this.id = id;
        this.label = label;
        this.blockSize = blockSize;
        this.typeSize = typeSize;
    }

    /** Bytes needed to store {@code numElements} elements of this type. */
    public long bytesFor(long numElements) {
        if (numElements % blockSize != 0) {
            throw new IllegalArgumentException(
                    "Element count " + numElements +
                            " is not a multiple of block size " + blockSize +
                            " for type " + label);
        }
        return (numElements / blockSize) * typeSize;
    }

    public static GgmlType fromId(int id) {
        for (GgmlType t : values()) {
            if (t.id == id)
                return t;
        }
        throw new IllegalArgumentException("Unknown ggml_type id: " + id);
    }

    public static GgmlType fromLabel(String label) {
        for (GgmlType t : values()) {
            if (t.label.equalsIgnoreCase(label))
                return t;
        }
        throw new IllegalArgumentException("Unknown ggml_type label: " + label);
    }

    /** True for floating-point types that are losslessly convertible. */
    public boolean isFloat() {
        return this == F32 || this == F16 || this == BF16 || this == F64;
    }
}
