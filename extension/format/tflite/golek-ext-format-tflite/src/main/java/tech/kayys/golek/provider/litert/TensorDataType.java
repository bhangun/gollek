package tech.kayys.golek.provider.litert;

public enum TensorDataType {
    FLOAT32(4),
    INT32(4),
    UINT32(4),
    UINT64(8),
    INT64(8),
    FLOAT64(8),
    UINT16(2),
    INT16(2),
    FLOAT16(2),
    BOOL(1),
    UINT8(1),
    INT8(1),
    STRING(1);

    private final int byteSize;

    TensorDataType(int byteSize) {
        this.byteSize = byteSize;
    }

    public int getByteSize() {
        return byteSize;
    }
}
