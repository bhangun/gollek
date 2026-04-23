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

    public TensorData dequantize(java.lang.foreign.Arena arena, long numElements) {
        if (isF32()) return this;
        MemorySegment f32 = arena.allocate(numElements * 4, 64);
        if (isF16()) {
            GGUFDequantizer.dequantizeF16(segment, f32, numElements);
        } else if (isQ8_0()) {
            GGUFDequantizer.dequantizeQ8_0(segment, 0, f32, numElements);
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + typeId);
        }
        return new TensorData(f32, 0);
    }
}
