package tech.kayys.gollek.quantizer.gptq;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Manages off-heap memory allocation for GPTQ tensor buffers using FFM API.
 *
 * Benefits over heap arrays:
 * - No GC pressure — tensor data lives off-heap
 * - SIMD-friendly: aligned allocations for Vector API operations
 * - Direct hardware access without JNI marshaling
 * - Deterministic release via Arena.close()
 *
 * Uses a single shared Arena for all allocations in a loading session,
 * ensuring all tensor buffers are released together.
 */
public class MemoryAllocator implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(MemoryAllocator.class);

    /**
     * Alignment in bytes for SIMD-friendly allocations (512-bit AVX-512 = 64 bytes)
     */
    public static final long SIMD_ALIGNMENT = 64L;

    /** Alignment for standard cache-line-aligned allocations */
    public static final long CACHE_LINE_ALIGNMENT = 64L;

    private final Arena arena;
    private long totalAllocated = 0L;
    private final List<String> allocationLog = new ArrayList<>();

    public MemoryAllocator() {
        // Shared arena allows segments to be passed across threads
        this.arena = Arena.ofAuto();
        log.debug("Created shared FFM Arena for tensor allocation");
    }

    /**
     * Allocates an off-heap buffer for INT32 data (qweight, qzeros, g_idx).
     *
     * @param count number of int32 elements
     * @param label debug label for tracking
     * @return a zero-initialized MemorySegment of count * 4 bytes
     */
    public MemorySegment allocateInt32(long count, String label) {
        long byteSize = count * Integer.BYTES;
        MemorySegment seg = allocateAligned(byteSize, SIMD_ALIGNMENT, label);
        log.debug("Allocated INT32 buffer '{}': {} elements, {} bytes", label, count, byteSize);
        return seg;
    }

    /**
     * Allocates an off-heap buffer for FP16 (short) data (scales, qzeros).
     *
     * @param count number of FP16 (short) elements
     * @param label debug label
     * @return a zero-initialized MemorySegment of count * 2 bytes
     */
    public MemorySegment allocateFp16(long count, String label) {
        long byteSize = count * Short.BYTES;
        MemorySegment seg = allocateAligned(byteSize, SIMD_ALIGNMENT, label);
        log.debug("Allocated FP16 buffer '{}': {} elements, {} bytes", label, count, byteSize);
        return seg;
    }

    /**
     * Allocates an off-heap buffer for FP32 data (dequantized output, bias).
     *
     * @param count number of float elements
     * @param label debug label
     * @return a zero-initialized MemorySegment of count * 4 bytes
     */
    public MemorySegment allocateFloat32(long count, String label) {
        long byteSize = count * Float.BYTES;
        MemorySegment seg = allocateAligned(byteSize, SIMD_ALIGNMENT, label);
        log.debug("Allocated FP32 buffer '{}': {} elements, {} bytes", label, count, byteSize);
        return seg;
    }

    /**
     * Core aligned allocation via FFM Arena.
     * All memory is zero-initialized by the JVM runtime.
     */
    public MemorySegment allocateAligned(long byteSize, long alignment, String label) {
        MemorySegment seg = arena.allocate(byteSize, alignment);
        totalAllocated += byteSize;
        allocationLog.add("[%s] %d bytes @ 0x%x".formatted(label, byteSize, seg.address()));
        return seg;
    }

    /**
     * Copies raw bytes from a source segment into a freshly allocated
     * aligned off-heap buffer.
     *
     * @param source source MemorySegment (e.g., mmap slice from parser)
     * @param label  debug label
     * @return new aligned segment containing a copy of the source data
     */
    public MemorySegment copyFrom(MemorySegment source, String label) {
        long size = source.byteSize();
        MemorySegment dest = allocateAligned(size, SIMD_ALIGNMENT, label);
        MemorySegment.copy(source, 0, dest, 0, size);
        return dest;
    }

    /**
     * Writes an int[] into a new off-heap segment.
     * Useful for transferring parsed arrays to off-heap SIMD buffers.
     */
    public MemorySegment fromInt32Array(int[] data, String label) {
        MemorySegment seg = allocateInt32(data.length, label);
        for (int i = 0; i < data.length; i++) {
            seg.set(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    (long) i * Integer.BYTES, data[i]);
        }
        return seg;
    }

    /**
     * Writes a short[] (FP16) into a new off-heap segment.
     */
    public MemorySegment fromFp16Array(short[] data, String label) {
        MemorySegment seg = allocateFp16(data.length, label);
        for (int i = 0; i < data.length; i++) {
            seg.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    (long) i * Short.BYTES, data[i]);
        }
        return seg;
    }

    /**
     * Writes a float[] (FP32) into a new off-heap segment.
     */
    public MemorySegment fromFloat32Array(float[] data, String label) {
        MemorySegment seg = allocateFloat32(data.length, label);
        for (int i = 0; i < data.length; i++) {
            seg.set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    (long) i * Float.BYTES, data[i]);
        }
        return seg;
    }

    // ── Read Helpers ──────────────────────────────────────────────────────────

    public static int readInt32(MemorySegment seg, long elementIndex) {
        return seg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                elementIndex * Integer.BYTES);
    }

    public static short readFp16(MemorySegment seg, long elementIndex) {
        return seg.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                elementIndex * Short.BYTES);
    }

    public static float readFloat32(MemorySegment seg, long elementIndex) {
        return seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                elementIndex * Float.BYTES);
    }

    public static void writeFloat32(MemorySegment seg, long elementIndex, float value) {
        seg.set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                elementIndex * Float.BYTES, value);
    }

    // ── FP16 ↔ FP32 Conversion ────────────────────────────────────────────────

    /**
     * Converts a raw FP16 bit pattern (stored as short) to FP32.
     *
     * FP16 layout: [s][eeeee][ffffffffff] (1 + 5 + 10 bits)
     * FP32 layout: [s][eeeeeeee][fffffffffffffffffffffff] (1 + 8 + 23 bits)
     *
     * This is the standard software FP16→FP32 expansion used in GPTQ dequant.
     */
    public static float fp16ToFloat32(short fp16) {
        int h = fp16 & 0xFFFF;
        int sign = (h >> 15) & 0x1;
        int exponent = (h >> 10) & 0x1F;
        int mantissa = h & 0x3FF;

        int fBits;
        if (exponent == 0) {
            if (mantissa == 0) {
                // ±0
                fBits = sign << 31;
            } else {
                // Denormalized FP16 → normalized FP32
                exponent = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x3FF;
                fBits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }
        } else if (exponent == 0x1F) {
            // Inf or NaN
            fBits = (sign << 31) | 0x7F800000 | (mantissa << 13);
        } else {
            // Normal FP16 → FP32: re-bias exponent (15 → 127)
            fBits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }

        return Float.intBitsToFloat(fBits);
    }

    /**
     * Converts a FP32 value to FP16 bit pattern (stored as short).
     * Used during output conversion.
     */
    public static short float32ToFp16(float f) {
        int bits = Float.floatToIntBits(f);
        int sign = (bits >> 31) & 0x1;
        int exponent = (bits >> 23) & 0xFF;
        int mantissa = bits & 0x7FFFFF;

        if (exponent == 0xFF) {
            // Inf or NaN
            return (short) ((sign << 15) | 0x7C00 | (mantissa != 0 ? 0x200 : 0));
        }

        int newExp = exponent - 127 + 15;
        if (newExp >= 31) {
            // Overflow → Inf
            return (short) ((sign << 15) | 0x7C00);
        }
        if (newExp <= 0) {
            // Underflow → denorm or zero
            if (newExp < -10)
                return (short) (sign << 15);
            mantissa = (mantissa | 0x800000) >> (1 - newExp);
            return (short) ((sign << 15) | (mantissa >> 13));
        }

        return (short) ((sign << 15) | (newExp << 10) | (mantissa >> 13));
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    public long getTotalAllocated() {
        return totalAllocated;
    }

    public void printStats() {
        System.out.printf("MemoryAllocator: %d allocations, total = %.2f MB%n",
                allocationLog.size(), totalAllocated / (1024.0 * 1024));
    }

    @Override
    public void close() {
        if (arena.scope().isAlive()) {
            log.info("Releasing FFM Arena: %.2f MB freed".formatted(totalAllocated / (1024.0 * 1024)));
            arena.close();
        }
    }
}
