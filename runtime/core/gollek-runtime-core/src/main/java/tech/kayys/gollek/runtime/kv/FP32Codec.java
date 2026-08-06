package tech.kayys.gollek.runtime.kv;

import java.lang.foreign.MemorySegment;

public final class FP32Codec implements KVCodec {
    @Override
    public void encode(MemorySegment src, MemorySegment dst, int elements) {
        MemorySegment.copy(src, 0, dst, 0, (long) elements * 4);
    }

    @Override
    public void decode(MemorySegment src, MemorySegment dst, int elements) {
        MemorySegment.copy(src, 0, dst, 0, (long) elements * 4);
    }

    @Override
    public int bytesPerElement() {
        return 4;
    }
}