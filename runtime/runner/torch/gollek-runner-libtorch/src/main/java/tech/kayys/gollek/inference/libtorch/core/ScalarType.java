package tech.kayys.gollek.inference.libtorch.core;

/**
 * Scalar type enumeration matching PyTorch dtypes.
 * Maps directly to at::ScalarType C++ enum values.
 */
public enum ScalarType {
    BYTE(0, "uint8", 1),
    CHAR(1, "int8", 1),
    SHORT(2, "int16", 2),
    INT(3, "int32", 4),
    LONG(4, "int64", 8),
    HALF(5, "float16", 2),
    FLOAT(6, "float32", 4),
    DOUBLE(7, "float64", 8),
    COMPLEX_HALF(8, "complex32", 4),
    COMPLEX_FLOAT(9, "complex64", 8),
    COMPLEX_DOUBLE(10, "complex128", 16),
    BOOL(11, "bool", 1),
    QINT8(12, "qint8", 1),
    QUINT8(13, "quint8", 1),
    QINT32(14, "qint32", 4),
    BFLOAT16(15, "bfloat16", 2);

    private final int code;
    private final String name;
    private final int byteSize;

    ScalarType(int code, String name, int byteSize) {
        this.code = code;
        this.name = name;
        this.byteSize = byteSize;
    }

    /** Native type code matching at::ScalarType. */
    public int code() {
        return code;
    }

    /** Human-readable type name. */
    public String typeName() {
        return name;
    }

    /** Size in bytes per element. */
    public int byteSize() {
        return byteSize;
    }

    /** Check if this is a floating-point type. */
    public boolean isFloatingPoint() {
        return this == HALF || this == FLOAT || this == DOUBLE || this == BFLOAT16;
    }

    /** Check if this is a complex type. */
    public boolean isComplex() {
        return this == COMPLEX_HALF || this == COMPLEX_FLOAT || this == COMPLEX_DOUBLE;
    }

    /** Check if this is a quantized type. */
    public boolean isQuantized() {
        return this == QINT8 || this == QUINT8 || this == QINT32;
    }

    /**
     * Look up ScalarType by its native code.
     *
     * @param code the at::ScalarType ordinal
     * @return the corresponding ScalarType
     * @throws IllegalArgumentException if code is unknown
     */
    public static ScalarType fromCode(int code) {
        for (ScalarType type : values()) {
            if (type.code == code) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown scalar type code: " + code);
    }

    /**
     * Map a safetensors dtype string to its ScalarType.
     * <p>
     * Safetensors uses uppercase abbreviations:
     * F32, F16, BF16, F64, I8, I16, I32, I64, U8, BOOL.
     *
     * @param dtype the safetensors dtype string
     * @return the corresponding ScalarType
     * @throws IllegalArgumentException if dtype is unknown
     */
    public static ScalarType fromSafetensors(String dtype) {
        if (dtype == null)
            throw new IllegalArgumentException("Null dtype");
        return switch (dtype.toUpperCase()) {
            case "F32" -> FLOAT;
            case "F64" -> DOUBLE;
            case "F16" -> HALF;
            case "BF16" -> BFLOAT16;
            case "I8" -> CHAR;
            case "I16" -> SHORT;
            case "I32" -> INT;
            case "I64" -> LONG;
            case "U8" -> BYTE;
            case "BOOL" -> BOOL;
            default -> throw new IllegalArgumentException("Unknown safetensors dtype: " + dtype);
        };
    }
}
