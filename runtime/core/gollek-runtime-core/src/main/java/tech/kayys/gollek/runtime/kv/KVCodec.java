package tech.kayys.gollek.runtime.kv;

import java.lang.foreign.MemorySegment;

public interface KVCodec {
    void encode(
            MemorySegment src, // FP32 input
            MemorySegment dst, // encoded storage
            int elements);

    void decode(
            MemorySegment src, // encoded storage
            MemorySegment dst, // FP32 output
            int elements);

    int bytesPerElement();
}