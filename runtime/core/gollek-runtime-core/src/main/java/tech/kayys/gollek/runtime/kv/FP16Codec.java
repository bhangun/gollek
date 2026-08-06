package tech.kayys.gollek.runtime.kv;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class FP16Codec implements KVCodec {
    @Override
    public void encode(MemorySegment src, MemorySegment dst, int elements) {
        for (int i = 0; i < elements; i++) {
            float v = src.get(ValueLayout.JAVA_FLOAT, (long) i * 4);
            short h = floatToHalf(v);
            dst.set(ValueLayout.JAVA_SHORT, (long) i * 2, h);
        }
    }

    @Override
    public void decode(MemorySegment src, MemorySegment dst, int elements) {
        for (int i = 0; i < elements; i++) {
            short h = src.get(ValueLayout.JAVA_SHORT, (long) i * 2);
            float v = halfToFloat(h);
            dst.set(ValueLayout.JAVA_FLOAT, (long) i * 4, v);
        }
    }

    @Override
    public int bytesPerElement() {
        return 2;
    }

    private short floatToHalf(float f) {
        int bits = Float.floatToIntBits(f);
        return (short) (bits >> 16);
    }

    private float halfToFloat(short h) {
        int bits = h << 16;
        return Float.intBitsToFloat(bits);
    }
}