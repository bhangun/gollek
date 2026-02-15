package io.github.pytorch.core;

/**
 * Scalar type enumeration matching PyTorch dtypes
 */
public enum ScalarType {
    BYTE(0),
    CHAR(1),
    SHORT(2),
    INT(3),
    LONG(4),
    HALF(5),
    FLOAT(6),
    DOUBLE(7),
    COMPLEX_HALF(8),
    COMPLEX_FLOAT(9),
    COMPLEX_DOUBLE(10),
    BOOL(11),
    QINT8(12),
    QUINT8(13),
    QINT32(14),
    BFLOAT16(15);
    
    private final int code;
    
    ScalarType(int code) {
        this.code = code;
    }
    
    public int code() {
        return code;
    }
    
    public static ScalarType fromCode(int code) {
        for (ScalarType type : values()) {
            if (type.code == code) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown scalar type code: " + code);
    }
}
