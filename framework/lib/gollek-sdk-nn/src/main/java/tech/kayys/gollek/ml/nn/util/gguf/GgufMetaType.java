package tech.kayys.gollek.ml.nn.util.gguf;

/**
 * Metadata value types in GGUF.
 */
public enum GgufMetaType {
    UINT8(0),
    INT8(1),
    UINT16(2),
    INT16(3),
    UINT32(4),
    INT32(5),
    FLOAT32(6),
    BOOL(7),
    STRING(8),
    ARRAY(9),
    UINT64(10),
    INT64(11),
    FLOAT64(12);

    public final int id;

    GgufMetaType(int id) {
        this.id = id;
    }

    public static GgufMetaType fromId(int id) {
        for (GgufMetaType t : values()) {
            if (t.id == id) return t;
        }
        throw new IllegalArgumentException("Unknown GGUF meta type id: " + id);
    }
}
