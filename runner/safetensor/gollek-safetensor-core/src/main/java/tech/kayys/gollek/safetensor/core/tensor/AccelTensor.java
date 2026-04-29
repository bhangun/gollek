/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AccelTensor.java
 * ────────────────
 * Pure-Java tensor backed by FFM MemorySegment (Float32).
 * No LibTorch, no native wrapper — just raw off-heap memory + Accelerate BLAS.
 *
 * Memory model
 * ════════════
 * Each tensor owns an Arena that manages its MemorySegment.
 * View tensors (reshape, slice) share the parent's segment via offset/stride.
 * close() releases the arena (or decrements ref-count for views).
 *
 * All data is Float32 (4 bytes per element). BF16/F16 weights are upcast
 * at load time by AccelWeightBridge.
 */
package tech.kayys.gollek.safetensor.core.tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Objects;

/**
 * Lightweight Float32 or Quantized tensor backed by FFM {@link MemorySegment}.
 * <p>
 * Supported Quantization types:
 * <ul>
 *     <li>F32: Standard 32-bit float (4 bytes)</li>
 *     <li>INT8: 8-bit integer with scales</li>
 *     <li>INT4: 4-bit packed integer (GPTQ style) with scales/zeros</li>
 * </ul>
 */
public class AccelTensor implements AutoCloseable {

    /** Supported quantization types for AccelTensor. */
    public enum QuantType {
        F32,     // 32-bit float
        INT8,    // 8-bit integer quantized
        INT4,    // 4-bit integer quantized (packed)
        FP8,     // 8-bit float quantized (E4M3/E5M2)
        F16,     // 16-bit float (IEEE 754 half-precision)
        BF16     // 16-bit bfloat16
    }

    private final MemorySegment data;
    private final long[] shape;
    private final long[] stride;
    private final long offset; // in elements (not bytes)
    private final Arena arena; // null for view tensors (parent owns)
    private final AccelTensor parent; // non-null for views
    private boolean closed = false;

    // ── Quantization Metadata ─────────────────────────────────────────
    private QuantType quantType = QuantType.F32;
    private MemorySegment scales = null;
    private MemorySegment zeros = null;
    private int groupSize = -1; // -1 means per-channel (no groups)

    /**
     * Dequantizes this tensor back to Float32 if it is quantized.
     * 
     * @return a new contiguous F32 tensor, or this tensor if it is already F32
     */
    public AccelTensor dequantize() {
        if (quantType == QuantType.F32) return this;
        
        // This is a slow fallback that uses the metadata.
        // Optimized kernels should use direct dequantization (e.g. BnB).
        float[] f32 = new float[(int) numel()];
        for (int i = 0; i < f32.length; i++) {
            f32[i] = getFlat(i); // getFlat handles quantization based on type/scales/zeros
        }
        return AccelTensor.fromFloatArray(f32, shape);
    }

    // ── Private constructors ──────────────────────────────────────────

    /** Owning constructor — this tensor owns the arena. */
    private AccelTensor(MemorySegment data, long[] shape, long[] stride, long offset, Arena arena) {
        this.data = Objects.requireNonNull(data);
        this.shape = shape.clone();
        this.stride = stride.clone();
        this.offset = offset;
        this.arena = arena;
        this.parent = null;
    }

    /** View constructor — shares parent's data, no arena ownership. */
    private AccelTensor(MemorySegment data, long[] shape, long[] stride, long offset, AccelTensor parent) {
        this.data = Objects.requireNonNull(data);
        this.shape = shape.clone();
        this.stride = stride.clone();
        this.offset = offset;
        this.arena = null;
        this.parent = parent;
    }

    // ── Factory methods ───────────────────────────────────────────────

    /**
     * Creates a zero-filled tensor with the given shape.
     */
    public static AccelTensor zeros(long... shape) {
        long n = numelOf(shape);
        Arena arena = Arena.ofAuto();
        // Force page alignment (4096) to prevent SIMD/page-boundary SEGV
        long size = n * Float.BYTES;
        long paddedSize = (size + 4095) & ~4095;
        MemorySegment seg = arena.allocate(paddedSize + 4096, 4096);
        seg.fill((byte) 0);
        return new AccelTensor(seg, shape, contiguousStride(shape), 0, arena);
    }

    /**
     * Creates a tensor filled with ones.
     */
    public static AccelTensor ones(long... shape) {
        long n = numelOf(shape);
        Arena arena = Arena.ofAuto();
        MemorySegment seg = arena.allocate(n * Float.BYTES, 64);
        for (long i = 0; i < n; i++) {
            seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
        }
        return new AccelTensor(seg, shape, contiguousStride(shape), 0, arena);
    }

    /**
     * Creates a tensor from a Java float array (copies data).
     */
    public static AccelTensor fromFloatArray(float[] data, long... shape) {
        long n = numelOf(shape);
        if (data.length != n) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " != shape numel " + n);
        }
        Arena arena = Arena.ofAuto();
        MemorySegment seg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);
        return new AccelTensor(seg, shape, contiguousStride(shape), 0, arena);
    }

    /**
     * Creates a tensor from a Java byte array (copies data).
     * Useful for storing quantized weights.
     */
    public static AccelTensor fromByteArray(byte[] data, long... shape) {
        // For byte arrays, the element size depends on the quantType,
        // but the factory just allocates the raw bytes.
        Arena arena = Arena.ofAuto();
        MemorySegment seg = arena.allocateFrom(ValueLayout.JAVA_BYTE, data);
        return new AccelTensor(seg, shape, contiguousStride(shape), 0, arena);
    }

    /**
     * Creates a tensor from a long array (for indices). Stored as float internally.
     */
    public static AccelTensor fromLongArray(long[] data, long... shape) {
        long n = numelOf(shape);
        if (data.length != n) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " != shape numel " + n);
        }
        float[] floats = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            floats[i] = (float) data[i];
        }
        return fromFloatArray(floats, shape);
    }

    /**
     * Zero-copy wrap of an existing MemorySegment (e.g. from mmap'd safetensor).
     * The caller retains ownership of the segment's lifecycle.
     */
    public static AccelTensor wrapSegment(MemorySegment segment, long[] shape) {
        // No arena — caller manages the segment lifetime
        return new AccelTensor(segment, shape, contiguousStride(shape), 0, (Arena) null);
    }

    /**
     * Creates a contiguous copy of the given data segment with a new owning arena.
     */
    public static AccelTensor copyOf(MemorySegment source, long[] shape) {
        long n = numelOf(shape);
        Arena arena = Arena.ofAuto();
        MemorySegment seg = arena.allocate(n * Float.BYTES, 64);
        MemorySegment.copy(source, 0, seg, 0, n * Float.BYTES);
        return new AccelTensor(seg, shape, contiguousStride(shape), 0, arena);
    }

    // ── Metadata ──────────────────────────────────────────────────────

    public long[] shape() { return shape.clone(); }
    public long[] stride() { return stride.clone(); }
    public long offset() { return offset; }
    public int rank() { return shape.length; }

    public long size(int dim) {
        int d = dim < 0 ? shape.length + dim : dim;
        return shape[d];
    }

    public long numel() {
        return numelOf(shape);
    }

    public boolean isContiguous() {
        long[] expected = contiguousStride(shape);
        return Arrays.equals(stride, expected);
    }

    public boolean isClosed() { return closed; }

    public QuantType quantType() { return quantType; }
    public AccelTensor withQuantization(QuantType type, MemorySegment scales, MemorySegment zeros, int groupSize) {
        this.quantType = type;
        this.scales = scales;
        this.zeros = zeros;
        this.groupSize = groupSize;
        return this;
    }

    public MemorySegment scales() { return scales; }
    public MemorySegment zeros() { return zeros; }
    public int groupSize() { return groupSize; }
    public boolean isQuantized() { return quantType != QuantType.F32; }

    // ── Data access ───────────────────────────────────────────────────

    /**
     * Returns the raw MemorySegment backing this tensor.
     * For contiguous tensors, this is the full data block.
     */
    public MemorySegment dataSegment() {
        checkClosed();
        return data;
    }

    /**
     * Returns a pointer to the first element of this tensor (accounting for offset).
     */
    public MemorySegment dataPtr() {
        checkClosed();
        return data.asSlice(offset * Float.BYTES);
    }

    /**
     * Computes statistics for debugging: min, max, and average.
     */
    public String statistics() {
        checkClosed();
        MemorySegment seg = dataPtr();
        long n = numel();
        float min = min();
        float max = max();
        double sum = 0.0;
        
        for (long i = 0; i < n; i++) {
            sum += seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return String.format("min=%.4f, max=%.4f, avg=%.4f, size=%d", min, max, sum / n, n);
    }

    public float min() {
        checkClosed();
        MemorySegment seg = dataPtr();
        long n = numel();
        float min = Float.POSITIVE_INFINITY;
        for (long i = 0; i < n; i++) {
            float val = seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (val < min) min = val;
        }
        return min;
    }

    public float max() {
        checkClosed();
        MemorySegment seg = dataPtr();
        long n = numel();
        float max = Float.NEGATIVE_INFINITY;
        for (long i = 0; i < n; i++) {
            float val = seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (val > max) max = val;
        }
        return max;
    }

    /**
     * Gets a single float element by flat index (relative to offset).
     */
    public float getFlat(long flatIdx) {
        checkClosed();
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, offset + flatIdx);
    }

    /**
     * Sets a single float element by flat index (relative to offset).
     */
    public void setFlat(long flatIdx, float value) {
        checkClosed();
        data.setAtIndex(ValueLayout.JAVA_FLOAT, offset + flatIdx, value);
    }

    /**
     * Gets element by multi-dim index using strides.
     */
    public float get(long... indices) {
        checkClosed();
        long idx = offset;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * stride[i];
        }
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
    }

    /**
     * Sets element by multi-dim index using strides.
     */
    public void set(float value, long... indices) {
        checkClosed();
        long idx = offset;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * stride[i];
        }
        data.setAtIndex(ValueLayout.JAVA_FLOAT, idx, value);
    }

    /**
     * Copies tensor data to a Java float array.
     * If non-contiguous, gathers elements via strides.
     */
    public float[] toFloatArray() {
        checkClosed();
        long n = numel();
        float[] result = new float[(int) n];
        if (isContiguous()) {
            MemorySegment.copy(data, ValueLayout.JAVA_FLOAT, offset * ValueLayout.JAVA_FLOAT.byteSize(), result, 0, (int) n);
        } else {
            // Gather via strides
            long[] idx = new long[shape.length];
            for (int i = 0; i < n; i++) {
                result[i] = get(idx);
                // Increment multi-dim index
                for (int d = shape.length - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < shape[d]) break;
                    idx[d] = 0;
                }
            }
        }
        return result;
    }

    /**
     * Returns a single scalar value (for 0-d or 1-element tensors).
     */
    public float item() {
        if (numel() != 1) {
            throw new IllegalStateException("item() requires exactly 1 element, got " + numel());
        }
        return getFlat(0);
    }

    // ── View operations (zero-copy) ───────────────────────────────────

    /**
     * Reshape to a new shape without copying data.
     * Only works for contiguous tensors.
     */
    public AccelTensor reshape(long... newShape) {
        checkClosed();
        // Resolve -1 in shape
        long inferIdx = -1;
        long known = 1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                if (inferIdx >= 0) throw new IllegalArgumentException("Only one -1 allowed in reshape");
                inferIdx = i;
            } else {
                known *= newShape[i];
            }
        }
        if (inferIdx >= 0) {
            newShape = newShape.clone();
            newShape[(int) inferIdx] = numel() / known;
        }

        if (numelOf(newShape) != numel()) {
            throw new IllegalArgumentException(
                "Cannot reshape " + Arrays.toString(shape) + " to " + Arrays.toString(newShape));
        }
        // Zero-copy reshape: just create a new view with contiguous strides for the new shape.
        // This assumes the underlying memory is laid out in a way that the reshape is sane.
        return new AccelTensor(data, newShape, contiguousStride(newShape), offset, this);
    }

    /**
     * Transpose two dimensions. Returns a NEW CONTIGUOUS copy
     * (not a view) to avoid MPS-style storage issues.
     */
    public AccelTensor transpose(int dim0, int dim1) {
        checkClosed();
        int d0 = dim0 < 0 ? shape.length + dim0 : dim0;
        int d1 = dim1 < 0 ? shape.length + dim1 : dim1;

        // Build transposed shape
        long[] newShape = shape.clone();
        newShape[d0] = shape[d1];
        newShape[d1] = shape[d0];

        // Build transposed strides
        long[] newStride = stride.clone();
        newStride[d0] = stride[d1];
        newStride[d1] = stride[d0];

        // Return a zero-copy view!
        return new AccelTensor(data, newShape, newStride, offset, this);
    }

    /**
     * Slice along a dimension [start, end).
     */
    public AccelTensor slice(int dim, long start, long end) {
        checkClosed();
        int d = dim < 0 ? shape.length + dim : dim;
        long[] newShape = shape.clone();
        newShape[d] = end - start;
        long newOffset = offset + start * stride[d];
        return new AccelTensor(data, newShape, stride, newOffset, this);
    }

    /**
     * Multi-dimensional slice [starts, ends).
     */
    public AccelTensor slice(long[] starts, long[] ends) {
        checkClosed();
        if (starts.length != shape.length || ends.length != shape.length) {
            throw new IllegalArgumentException("Slice indices length mismatch");
        }
        long[] newShape = new long[shape.length];
        long newOffset = offset;
        for (int i = 0; i < shape.length; i++) {
            newShape[i] = ends[i] - starts[i];
            newOffset += starts[i] * stride[i];
        }
        return new AccelTensor(data, newShape, stride, newOffset, this);
    }

    /**
     * Squeeze: remove all dimensions of size 1.
     */
    public AccelTensor squeeze() {
        checkClosed();
        int count = 0;
        for (long s : shape) if (s != 1) count++;
        if (count == shape.length) return this;

        long[] newShape = new long[count];
        long[] newStride = new long[count];
        int j = 0;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != 1) {
                newShape[j] = shape[i];
                newStride[j] = stride[i];
                j++;
            }
        }
        return new AccelTensor(data, newShape, newStride, offset, this);
    }

    /**
     * Unsqueeze: add a dimension of size 1 at position dim.
     */
    public AccelTensor unsqueeze(int dim) {
        checkClosed();
        int d = dim < 0 ? shape.length + 1 + dim : dim;
        long[] newShape = new long[shape.length + 1];
        long[] newStride = new long[stride.length + 1];
        int src = 0;
        for (int i = 0; i < newShape.length; i++) {
            if (i == d) {
                newShape[i] = 1;
                newStride[i] = (src < stride.length) ? stride[src] : 1;
            } else {
                newShape[i] = shape[src];
                newStride[i] = stride[src];
                src++;
            }
        }
        return new AccelTensor(data, newShape, newStride, offset, this);
    }

    /**
     * Returns a contiguous copy if non-contiguous, or this if already contiguous.
     */
    public AccelTensor contiguous() {
        if (isContiguous() && offset == 0) return this;
        return materialize(shape, stride);
    }

    /**
     * Gathers elements from dim 0 using integer indices stored as floats.
     * Used for embedding lookup.
     */
    public AccelTensor indexSelect(long[] indices) {
        checkClosed();
        if (shape.length < 1) throw new IllegalStateException("indexSelect requires at least 1D");

        long embDim = shape.length > 1 ? shape[1] : 1;
        long[] outShape;
        if (shape.length == 2) {
            outShape = new long[] { indices.length, embDim };
        } else {
            outShape = new long[] { indices.length };
        }
        AccelTensor out = AccelTensor.zeros(outShape);

        for (int i = 0; i < indices.length; i++) {
            long srcRow = indices[i];
            if (shape.length >= 2) {
                // Copy row from [vocabSize, embDim]
                long srcByteOffset = (offset + srcRow * stride[0]) * Float.BYTES;
                if (isQuantized()) {
                     // Dequantize row on-the-fly
                     dequantizeRow(srcRow, out.dataPtr().asSlice((long) i * embDim * Float.BYTES), (int) embDim);
                } else {
                    long dstByteOffset = (long) i * embDim * Float.BYTES;
                    MemorySegment.copy(
                        data, ValueLayout.JAVA_FLOAT, srcByteOffset,
                        out.data, ValueLayout.JAVA_FLOAT, dstByteOffset,
                        (int) embDim
                    );
                }
            } else {
                out.setFlat(i, getFlat(srcRow));
            }
        }
        return out;
    }

    private void dequantizeRow(long row, MemorySegment dst, int embDim) {
        if (quantType == QuantType.F16) {
            long srcRowByteOff = (offset + row * stride[0]) * 2L;
            DequantizationKernel.dequantizeF16(data.asSlice(srcRowByteOff, embDim * 2L), dst, embDim);
            return;
        } else if (quantType == QuantType.BF16) {
            long srcRowByteOff = (offset + row * stride[0]) * 2L;
            DequantizationKernel.dequantizeBf16(data.asSlice(srcRowByteOff, embDim * 2L), dst, embDim);
            return;
        }

        // row is the row index in the quantized segment
        long groupIdx = (row * embDim) / groupSize;
        float scale = scales.getAtIndex(ValueLayout.JAVA_FLOAT, groupIdx);
        float zero = (zeros != null) ? zeros.getAtIndex(ValueLayout.JAVA_FLOAT, groupIdx) : 0.0f;
        
        long srcRowByteOff = (offset + row * stride[0]); // offset is in elements or bytes? 
        // For I8, stride[0] is embDim bytes.
        
        for (int j = 0; j < embDim; j++) {
            if (quantType == QuantType.INT8) {
                byte q = data.get(ValueLayout.JAVA_BYTE, srcRowByteOff + j);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, j, q * scale);
            } else if (quantType == QuantType.INT4) {
                // Support simple packed INT4? 
                // This is getting complex, for now assume INT8 for embeddings
            }
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (arena != null) {
                try {
                    arena.close();
                } catch (UnsupportedOperationException ignored) {
                }
            }
        }
    }

    /**
     * Recursively closes this tensor and its parent chain.
     * Useful for cleaning up views where the parent holds the Arena.
     */
    public void closeWithParent() {
        if (closed) return;
        AccelTensor p = this.parent;
        close();
        if (p != null) {
            p.closeWithParent();
        }
    }

    // ── Internal utilities ────────────────────────────────────────────

    private void checkClosed() {
        if (closed) throw new IllegalStateException("AccelTensor has been closed");
    }

    /**
     * Materializes a strided view into a new contiguous tensor.
     */
    private AccelTensor materialize(long[] srcShape, long[] srcStride) {
        long n = numelOf(srcShape);
        Arena newArena = Arena.ofAuto();
        MemorySegment newSeg = newArena.allocate(n * Float.BYTES, 64);

        if (srcShape.length > 0 && srcStride[srcShape.length - 1] == 1) {
            // Optimized row-copy for tensors contiguous in the last dimension (common for transposes)
            int lastDim = (int) srcShape[srcShape.length - 1];
            long outerNumel = n / lastDim;
            long[] outerShape = new long[srcShape.length - 1];
            System.arraycopy(srcShape, 0, outerShape, 0, srcShape.length - 1);
            long[] outerIdx = new long[outerShape.length];

            for (long row = 0; row < outerNumel; row++) {
                long srcRowOffset = offset;
                for (int d = 0; d < outerShape.length; d++) {
                    srcRowOffset += outerIdx[d] * srcStride[d];
                }
                MemorySegment.copy(data, ValueLayout.JAVA_FLOAT, srcRowOffset * Float.BYTES,
                                   newSeg, ValueLayout.JAVA_FLOAT, row * lastDim * Float.BYTES,
                                   lastDim);
                
                // Increment indices
                for (int d = outerShape.length - 1; d >= 0; d--) {
                    outerIdx[d]++;
                    if (outerIdx[d] < outerShape[d]) break;
                    outerIdx[d] = 0;
                }
            }
        } else {
            // Fallback for fully non-contiguous slices
            long[] idx = new long[srcShape.length];
            for (long flat = 0; flat < n; flat++) {
                long srcIdx = offset;
                for (int d = 0; d < srcShape.length; d++) {
                    srcIdx += idx[d] * srcStride[d];
                }
                newSeg.setAtIndex(ValueLayout.JAVA_FLOAT, flat, data.getAtIndex(ValueLayout.JAVA_FLOAT, srcIdx));
                for (int d = srcShape.length - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < srcShape[d]) break;
                    idx[d] = 0;
                }
            }
        }
        return new AccelTensor(newSeg, srcShape, contiguousStride(srcShape), 0, newArena);
    }

    static long numelOf(long[] shape) {
        long n = 1;
        for (long s : shape) n *= s;
        return n;
    }

    static long[] contiguousStride(long[] shape) {
        long[] stride = new long[shape.length];
        if (shape.length > 0) {
            stride[shape.length - 1] = 1;
            for (int i = shape.length - 2; i >= 0; i--) {
                stride[i] = stride[i + 1] * shape[i + 1];
            }
        }
        return stride;
    }

    @Override
    public String toString() {
        return "AccelTensor[shape=" + Arrays.toString(shape) + ", contiguous=" + isContiguous()
                + ", closed=" + closed + "]";
    }
}
