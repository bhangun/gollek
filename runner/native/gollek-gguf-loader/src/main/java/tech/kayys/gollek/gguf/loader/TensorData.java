package tech.kayys.gollek.gguf.loader;

import java.lang.foreign.MemorySegment;

/**
 * Encapsulates a zero-copy memory segment and its quantization type.
 * Used for lazy dequantization during inference.
 */
public record TensorData(MemorySegment segment, int typeId) {
    public boolean isF32() { return typeId == 0; }
    public boolean isF16() { return typeId == 1; }
    public boolean isQ8_0() { return typeId == 8; }
}
