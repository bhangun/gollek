package tech.kayys.gollek.ml.nn.util.gguf;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * GGUF Metadata value container.
 */
public sealed interface GgufMetaValue {

    record Uint8Val(short value) implements GgufMetaValue {}
    record Int8Val(byte value) implements GgufMetaValue {}
    record Uint16Val(int value) implements GgufMetaValue {}
    record Int16Val(short value) implements GgufMetaValue {}
    record Uint32Val(long value) implements GgufMetaValue {}
    record Int32Val(int value) implements GgufMetaValue {}
    record Float32Val(float value) implements GgufMetaValue {}
    record Uint64Val(long value) implements GgufMetaValue {}
    record Int64Val(long value) implements GgufMetaValue {}
    record Float64Val(double value) implements GgufMetaValue {}
    record BoolVal(boolean value) implements GgufMetaValue {}
    record StringVal(String value) implements GgufMetaValue {}

    record ArrayVal(GgufMetaType type, List<GgufMetaValue> values) implements GgufMetaValue {
        public ArrayVal {
            values = Collections.unmodifiableList(new ArrayList<>(values));
        }
    }

    /**
     * Helper to wrap a string.
     */
    static GgufMetaValue ofString(String value) {
        return new StringVal(value);
    }

    /**
     * Helper to wrap a uint32.
     */
    static GgufMetaValue ofUint32(long value) {
        return new Uint32Val(value);
    }
}
