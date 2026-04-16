package tech.kayys.gollek.converter.gguf;

import java.util.List;

/**
 * Sealed hierarchy for typed GGUF metadata values.
 * Mirrors the {@code gguf_value} union from the C spec.
 */
public sealed interface GgufMetaValue
        permits GgufMetaValue.UInt8Val, GgufMetaValue.Int8Val,
        GgufMetaValue.UInt16Val, GgufMetaValue.Int16Val,
        GgufMetaValue.UInt32Val, GgufMetaValue.Int32Val,
        GgufMetaValue.Float32Val, GgufMetaValue.BoolVal,
        GgufMetaValue.StringVal, GgufMetaValue.ArrayVal,
        GgufMetaValue.UInt64Val, GgufMetaValue.Int64Val,
        GgufMetaValue.Float64Val {

    GgufMetaType type();

    record UInt8Val(short value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.UINT8;
        }
    }

    record Int8Val(byte value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.INT8;
        }
    }

    record UInt16Val(int value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.UINT16;
        }
    }

    record Int16Val(short value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.INT16;
        }
    }

    record UInt32Val(long value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.UINT32;
        }
    }

    record Int32Val(int value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.INT32;
        }
    }

    record Float32Val(float value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.FLOAT32;
        }
    }

    record BoolVal(boolean value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.BOOL;
        }
    }

    record StringVal(String value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.STRING;
        }
    }

    record UInt64Val(long value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.UINT64;
        }
    }

    record Int64Val(long value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.INT64;
        }
    }

    record Float64Val(double value) implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.FLOAT64;
        }
    }

    /**
     * Typed array – the element type is stored separately from the container.
     */
    record ArrayVal(GgufMetaType elementType, List<GgufMetaValue> elements)
            implements GgufMetaValue {
        public GgufMetaType type() {
            return GgufMetaType.ARRAY;
        }
    }

    // ── Convenience factories ──────────────────────────────────────────────

    static GgufMetaValue ofString(String s) {
        return new StringVal(s);
    }

    static GgufMetaValue ofUInt32(long v) {
        return new UInt32Val(v);
    }

    static GgufMetaValue ofInt32(int v) {
        return new Int32Val(v);
    }

    static GgufMetaValue ofFloat32(float v) {
        return new Float32Val(v);
    }

    static GgufMetaValue ofBool(boolean v) {
        return new BoolVal(v);
    }

    static GgufMetaValue ofUInt64(long v) {
        return new UInt64Val(v);
    }

    static GgufMetaValue ofStringArray(List<String> strs) {
        return new ArrayVal(
                GgufMetaType.STRING,
                strs.stream().<GgufMetaValue>map(StringVal::new).toList());
    }

    static GgufMetaValue ofInt32Array(List<Integer> ints) {
        return new ArrayVal(
                GgufMetaType.INT32,
                ints.stream().<GgufMetaValue>map(Int32Val::new).toList());
    }

    static GgufMetaValue ofFloat32Array(List<Float> floats) {
        return new ArrayVal(
                GgufMetaType.FLOAT32,
                floats.stream().<GgufMetaValue>map(Float32Val::new).toList());
    }

    // ── Typed accessors (throw on mismatch) ───────────────────────────────

    default String asString() {
        if (this instanceof StringVal sv)
            return sv.value();
        throw new ClassCastException("Not a STRING value: " + type());
    }

    default long asUInt32() {
        if (this instanceof UInt32Val v)
            return v.value();
        if (this instanceof Int32Val v)
            return Integer.toUnsignedLong(v.value());
        throw new ClassCastException("Not a UINT32 value: " + type());
    }

    default int asInt32() {
        if (this instanceof Int32Val v)
            return v.value();
        throw new ClassCastException("Not an INT32 value: " + type());
    }

    default float asFloat32() {
        if (this instanceof Float32Val v)
            return v.value();
        throw new ClassCastException("Not a FLOAT32 value: " + type());
    }

    default boolean asBool() {
        if (this instanceof BoolVal v)
            return v.value();
        throw new ClassCastException("Not a BOOL value: " + type());
    }

    default long asUInt64() {
        if (this instanceof UInt64Val v)
            return v.value();
        throw new ClassCastException("Not a UINT64 value: " + type());
    }

    default List<GgufMetaValue> asArray() {
        if (this instanceof ArrayVal av)
            return av.elements();
        throw new ClassCastException("Not an ARRAY value: " + type());
    }
}
