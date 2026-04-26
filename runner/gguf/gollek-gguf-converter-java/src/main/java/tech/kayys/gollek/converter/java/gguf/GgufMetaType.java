package tech.kayys.gollek.converter.java.gguf;

/**
 * GGUF metadata value types (gguf_type enum from the spec).
 * Values are little-endian uint32 on disk.
 */
public enum GgufMetaType {
    UINT8(0, 1),
    INT8(1, 1),
    UINT16(2, 2),
    INT16(3, 2),
    UINT32(4, 4),
    INT32(5, 4),
    FLOAT32(6, 4),
    BOOL(7, 1),
    STRING(8, -1), // variable length
    ARRAY(9, -1), // variable length
    UINT64(10, 8),
    INT64(11, 8),
    FLOAT64(12, 8);

    public final int id;
    public final int byteSize; // -1 = variable

    GgufMetaType(int id, int byteSize) {
        this.id = id;
        this.byteSize = byteSize;
    }

    public static GgufMetaType fromId(int id) {
        for (GgufMetaType t : values()) {
            if (t.id == id)
                return t;
        }
        throw new IllegalArgumentException("Unknown GGUF metadata type id: " + id);
    }
}
