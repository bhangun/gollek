package tech.kayys.gollek.runtime.kv;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class Int8Codec implements KVCodec {
    private final float scale;

    public Int8Codec(float scale) {
        this.scale = scale;
    }

    public Int8Codec() {
        this(127f); // default symmetric scale
    }

    @Override
    public void encode(MemorySegment src, MemorySegment dst, int elements) {
        for (int i = 0; i < elements; i++) {
            float v = src.get(ValueLayout.JAVA_FLOAT, (long) i * 4);
            int q = Math.round(v * scale);
            if (q > 127)
                q = 127;
            if (q < -128)
                q = -128;
            dst.set(ValueLayout.JAVA_BYTE, i, (byte) q);
        }
    }

    @Override
    public void decode(MemorySegment src, MemorySegment dst, int elements) {
        for (int i = 0; i < elements; i++) {
            byte q = src.get(ValueLayout.JAVA_BYTE, i);
            float v = q / scale;
            dst.set(ValueLayout.JAVA_FLOAT, (long) i * 4, v);
        }
    }

    @Override
    public int bytesPerElement() {
        return 1;
    }
}