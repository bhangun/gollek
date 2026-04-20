/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorTensor.java
 * ─────────────────────
 * Live, zero-copy view of a single tensor whose bytes live in an off-heap
 * {@link java.lang.foreign.MemorySegment} (JDK 25 FFM API, finalized in JEP 454).
 *
 * Design goals
 * ════════════
 * 1. ZERO COPY — the MemorySegment is a window into the mmap'd file; no heap
 *    allocation is required to access element data.
 * 2. SAFETY    — all bounds checks go through the FFM segment bounds checker,
 *    giving JVM-level safety without copying into arrays.
 * 3. LIFETIME  — the segment's Arena controls deallocation; callers should not
 *    hold a SafetensorTensor after its parent SafetensorLoadResult is closed.
 * 4. THREAD-SAFE READS — MemorySegment reads are safe from multiple threads
 *    when the underlying Arena is a shared arena.
 *
 * Byte ordering
 * ═════════════
 * SafeTensors stores all multi-byte values in LITTLE-ENDIAN order.
 * All typed accessors in this class explicitly use ByteOrder.LITTLE_ENDIAN.
 */
package tech.kayys.gollek.safetensor.loader;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Zero-copy, off-heap view of a single tensor in a SafeTensors file.
 *
 * <p>
 * The underlying bytes are stored in a {@link MemorySegment} (either
 * memory-mapped from disk or copied into native memory). All element
 * accesses go directly to the off-heap region — no heap buffer copy.
 *
 * <p>
 * <b>Lifetime contract:</b> this object must not be used after the
 * {@link tech.kayys.gollek.inference.safetensor.ffm.SafetensorLoadResult}
 * that produced it has been {@link AutoCloseable#close() closed}. Accessing
 * a closed segment causes a JVM-level exception from the FFM layer.
 */
public final class SafetensorTensor implements AutoCloseable {

    // ValueLayouts for little-endian element access
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfDouble DOUBLE_LE = ValueLayout.JAVA_DOUBLE.withOrder(ByteOrder.LITTLE_ENDIAN);

    // ─────────────────────────────────────────────────────────────────────────

    /** AccelTensor metadata (name, dtype, shape, offsets). */
    private final SafetensorTensorInfo info;

    /**
     * Off-heap segment that starts exactly at the first byte of this tensor's
     * data. The segment's byte-size equals {@code info.byteLength()}.
     */
    private final MemorySegment segment;

    /** {@code true} after {@link #close()} — guards against double-free. */
    private volatile boolean closed = false;

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Package-private constructor. Instantiated exclusively by
     * {@link tech.kayys.gollek.inference.safetensor.ffm.SafetensorFFMLoader}.
     *
     * @param info    tensor metadata
     * @param segment memory segment exactly spanning this tensor's bytes
     */
    SafetensorTensor(SafetensorTensorInfo info, MemorySegment segment) {
        this.info = Objects.requireNonNull(info, "info must not be null");
        this.segment = Objects.requireNonNull(segment, "segment must not be null");
    }

    // ── Metadata accessors ────────────────────────────────────────────────────

    /** AccelTensor name as in the file header. */
    public String name() {
        return info.name();
    }

    /** Element data type. */
    public SafetensorDType dtype() {
        return info.dtype();
    }

    /** AccelTensor dimensions (defensive copy). */
    public long[] shape() {
        return info.shape();
    }

    /** Number of dimensions. */
    public int rank() {
        return info.rank();
    }

    /** Total number of scalar elements. */
    public long numElements() {
        return info.numElements();
    }

    /** Byte length of the raw data (elements × dtype.byteSize()). */
    public long byteLength() {
        return info.byteLength();
    }

    /** Full tensor metadata descriptor. */
    public SafetensorTensorInfo info() {
        return info;
    }

    // ── Raw segment access ────────────────────────────────────────────────────

    /**
     * Return the raw {@link MemorySegment} backing this tensor.
     *
     * <p>
     * The segment starts at byte 0 of this tensor's data and has a
     * byte-size of {@code byteLength()}. The byte order of multi-byte
     * elements is little-endian.
     *
     * <p>
     * <b>WARNING:</b> the returned segment is only valid while the parent
     * {@link tech.kayys.gollek.inference.safetensor.ffm.SafetensorLoadResult}
     * is open.
     *
     * @return the raw off-heap segment
     * @throws IllegalStateException if this tensor (or its load result) is closed
     */
    public MemorySegment segment() {
        checkOpen();
        return segment;
    }

    // ── Copy-to-heap accessors (safe, bounds-checked) ─────────────────────────

    /**
     * Copy the entire tensor data into a freshly allocated {@code byte[]}.
     *
     * <p>
     * Useful for consumers that require a heap array (e.g. JNI, ONNX Runtime
     * Java bindings). For large tensors, prefer {@link #segment()} to avoid the
     * allocation.
     *
     * @return a new byte array containing the tensor bytes (little-endian)
     * @throws IllegalStateException if this tensor is closed
     * @throws OutOfMemoryError      if the tensor exceeds {@code Integer.MAX_VALUE}
     *                               bytes
     */
    public byte[] toByteArray() {
        checkOpen();
        long len = segment.byteSize();
        if (len > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException(
                    "AccelTensor '" + name() + "' is too large to fit in a Java array ("
                            + len + " bytes). Use segment() for direct memory access.");
        }
        return segment.toArray(ValueLayout.JAVA_BYTE);
    }

    /**
     * Copy the tensor data into a {@code float[]} — valid only for F32 tensors.
     *
     * @return float array of length {@link #numElements()}
     * @throws IllegalStateException         if the tensor is closed
     * @throws UnsupportedOperationException if the dtype is not F32
     */
    public float[] toFloatArray() {
        checkOpen();
        checkArrayFit();
        long n = numElements();
        if (dtype() == SafetensorDType.F32) {
            return segment.toArray(FLOAT_LE);
        } else if (dtype() == SafetensorDType.BF16) {
            float[] result = new float[(int) n];
            for (long i = 0; i < n; i++) {
                result[(int) i] = getBF16AsFloat(i);
            }
            return result;
        } else if (dtype() == SafetensorDType.F16) {
            float[] result = new float[(int) n];
            for (long i = 0; i < n; i++) {
                result[(int) i] = getF16AsFloat(i);
            }
            return result;
        }
        requireDtype(SafetensorDType.F32, SafetensorDType.BF16, SafetensorDType.F16);
        return null;
    }

    /**
     * Copy the tensor data into a {@code double[]} — valid only for F64 tensors.
     */
    public double[] toDoubleArray() {
        checkOpen();
        requireDtype(SafetensorDType.F64);
        checkArrayFit();
        return segment.toArray(DOUBLE_LE);
    }

    /**
     * Copy the tensor data into a {@code int[]} — valid only for I32 / U32.
     */
    public int[] toIntArray() {
        checkOpen();
        requireDtype(SafetensorDType.I32, SafetensorDType.U32);
        checkArrayFit();
        return segment.toArray(INT_LE);
    }

    /**
     * Copy the tensor data into a {@code long[]} — valid only for I64 / U64.
     */
    public long[] toLongArray() {
        checkOpen();
        requireDtype(SafetensorDType.I64, SafetensorDType.U64);
        checkArrayFit();
        return segment.toArray(LONG_LE);
    }

    /**
     * Copy the tensor data into a {@code short[]} — valid for F16, BF16,
     * I16, U16 (raw bits; BF16 requires caller-side conversion).
     */
    public short[] toShortArray() {
        checkOpen();
        requireDtype(SafetensorDType.F16, SafetensorDType.BF16,
                SafetensorDType.I16, SafetensorDType.U16);
        checkArrayFit();
        return segment.toArray(SHORT_LE);
    }

    // ── Element-level random access ───────────────────────────────────────────

    /**
     * Read a single F32 element at the given multi-dimensional indices.
     *
     * @param indices one index per dimension (row-major / C order)
     * @return the float value at those coordinates
     */
    public float getFloat(long... indices) {
        checkOpen();
        requireDtype(SafetensorDType.F32);
        long byteOffset = info.elementByteOffset(indices);
        return segment.get(FLOAT_LE, byteOffset);
    }

    /**
     * Read a single F64 element at the given multi-dimensional indices.
     */
    public double getDouble(long... indices) {
        checkOpen();
        requireDtype(SafetensorDType.F64);
        long byteOffset = info.elementByteOffset(indices);
        return segment.get(DOUBLE_LE, byteOffset);
    }

    /**
     * Read a single I32 / U32 element at the given multi-dimensional indices.
     */
    public int getInt(long... indices) {
        checkOpen();
        requireDtype(SafetensorDType.I32, SafetensorDType.U32);
        long byteOffset = info.elementByteOffset(indices);
        return segment.get(INT_LE, byteOffset);
    }

    /**
     * Read a single I64 / U64 element at the given multi-dimensional indices.
     */
    public long getLong(long... indices) {
        checkOpen();
        requireDtype(SafetensorDType.I64, SafetensorDType.U64);
        long byteOffset = info.elementByteOffset(indices);
        return segment.get(LONG_LE, byteOffset);
    }

    /**
     * Read raw bytes of a single element at the given multi-dimensional
     * indices. Works for any dtype. Returns a defensive copy of
     * {@code dtype.byteSize()} bytes.
     */
    public byte[] getElementBytes(long... indices) {
        checkOpen();
        long byteOffset = info.elementByteOffset(indices);
        int byteSize = info.dtype().byteSize();
        MemorySegment slice = segment.asSlice(byteOffset, byteSize);
        return slice.toArray(ValueLayout.JAVA_BYTE);
    }

    /**
     * Convert a BF16 element to its nearest F32 representation.
     * BF16 is stored as 2 bytes; this method sign-extends the exponent/mantissa
     * into the upper 16 bits of a float32.
     *
     * @param indices multi-dimensional indices
     * @return the value as a Java {@code float}
     */
    public float getBF16AsFloat(long... indices) {
        checkOpen();
        requireDtype(SafetensorDType.BF16);
        long byteOffset = info.elementByteOffset(indices);
        return getBF16AsFloatAtOffset(byteOffset);
    }

    /**
     * Read a BF16 element at a flat index.
     */
    public float getBF16AsFloat(long flatIndex) {
        checkOpen();
        requireDtype(SafetensorDType.BF16);
        return getBF16AsFloatAtOffset(flatIndex * 2);
    }

    private float getBF16AsFloatAtOffset(long byteOffset) {
        short rawBits = segment.get(SHORT_LE, byteOffset);
        // BF16 is the top 16 bits of F32 — shift left by 16
        int floatBits = (rawBits & 0xFFFF) << 16;
        return Float.intBitsToFloat(floatBits);
    }

    /**
     * Convert an F16 (IEEE 754 half-precision) element to its nearest F32.
     *
     * @param indices multi-dimensional indices
     * @return the value as a Java {@code float}
     */
    public float getF16AsFloat(long... indices) {
        checkOpen();
        requireDtype(SafetensorDType.F16);
        long byteOffset = info.elementByteOffset(indices);
        return getF16AsFloatAtOffset(byteOffset);
    }

    /**
     * Read an F16 element at a flat index.
     */
    public float getF16AsFloat(long flatIndex) {
        checkOpen();
        requireDtype(SafetensorDType.F16);
        return getF16AsFloatAtOffset(flatIndex * 2);
    }

    private float getF16AsFloatAtOffset(long byteOffset) {
        short rawBits = segment.get(SHORT_LE, byteOffset);
        return float16ToFloat32(rawBits);
    }

    // ── Slice ─────────────────────────────────────────────────────────────────

    /**
     * Return a slice of this tensor's backing segment as a new MemorySegment.
     * The slice is relative to the tensor's own byte offset 0.
     *
     * <p>
     * Useful for processing a sub-range of the tensor without copying.
     *
     * @param byteOffset start byte relative to this tensor's data start
     * @param byteLength number of bytes in the slice
     * @return a sub-segment sharing the same backing memory
     */
    public MemorySegment slice(long byteOffset, long byteLength) {
        checkOpen();
        return segment.asSlice(byteOffset, byteLength);
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Idempotent release. The segment's Arena controls the actual
     * deallocation; this only marks the tensor as logically closed so that
     * subsequent accesses fail fast.
     *
     * <p>
     * In normal use, tensors are closed implicitly when the owning
     * {@link tech.kayys.gollek.inference.safetensor.ffm.SafetensorLoadResult}
     * is closed.
     */
    @Override
    public void close() {
        closed = true;
    }

    public boolean isClosed() {
        return closed;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "AccelTensor '" + name() + "' has been closed. "
                            + "Ensure the parent SafetensorLoadResult is still open.");
        }
    }

    private void requireDtype(SafetensorDType... allowed) {
        for (SafetensorDType a : allowed) {
            if (info.dtype() == a)
                return;
        }
        throw new UnsupportedOperationException(
                "Operation requires dtype " + Arrays.toString(allowed)
                        + " but tensor '" + name() + "' has dtype " + info.dtype());
    }

    private void checkArrayFit() {
        if (numElements() > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException(
                    "AccelTensor '" + name() + "' has " + numElements()
                            + " elements — too large to copy into a Java array. "
                            + "Use segment() for zero-copy access.");
        }
    }

    /**
     * IEEE 754 half-precision (F16) to single-precision (F32) conversion.
     *
     * <p>
     * Handles subnormals, infinities and NaNs correctly.
     */
    private static float float16ToFloat32(short half) {
        int h = half & 0xFFFF;
        int sign = (h >> 15) & 0x1;
        int exponent = (h >> 10) & 0x1F;
        int mantissa = h & 0x3FF;

        int f32;
        if (exponent == 0) {
            if (mantissa == 0) {
                // +/- 0
                f32 = sign << 31;
            } else {
                // Subnormal F16 → normalised F32
                exponent = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= ~0x400;
                f32 = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
            }
        } else if (exponent == 31) {
            // Infinity or NaN
            f32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        } else {
            // Normalised
            f32 = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
        }
        return Float.intBitsToFloat(f32);
    }

    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public String toString() {
        return "SafetensorTensor{"
                + "name='" + name() + '\''
                + ", dtype=" + dtype().jsonName()
                + ", shape=" + Arrays.toString(shape())
                + ", bytes=" + byteLength()
                + ", closed=" + closed
                + '}';
    }
}
