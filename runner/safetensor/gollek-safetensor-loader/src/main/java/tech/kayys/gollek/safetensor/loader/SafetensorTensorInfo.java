/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorTensorInfo.java
 * ─────────────────────────
 * Immutable representation of one tensor's metadata block as decoded from the
 * SafeTensors JSON header.
 *
 * SafeTensors header JSON format (per tensor):
 * ┌───────────────────────────────────────────────────────────────────┐
 * │  "weight_name": {                                                 │
 * │      "dtype": "F32",                                              │
 * │      "shape": [4096, 4096],                                       │
 * │      "data_offsets": [0, 67108864]   // [begin, end) in data blob │
 * │  }                                                                │
 * └───────────────────────────────────────────────────────────────────┘
 *
 * Derived fields (byteLength, numElements) are computed lazily and cached
 * to avoid redundant arithmetic during hot-path tensor access.
 */
package tech.kayys.gollek.safetensor.loader;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable tensor descriptor decoded from a SafeTensors JSON header.
 *
 * <p>
 * All coordinate arithmetic (offset, length, element-count) is performed
 * in {@code long} to handle models whose weight tensors exceed 2 GiB.
 *
 * <p>
 * Instances are created by the JSON deserialiser inside
 * {@link SafetensorHeader} and are subsequently kept alive for the lifetime of
 * the loaded model. They are safe to share across threads.
 */
public final class SafetensorTensorInfo {

    /** TorchTensor name as-is from the JSON header key. */
    private final String name;

    /** Element data type. */
    private final SafetensorDType dtype;

    /**
     * TorchTensor dimensions, e.g. {@code [4096, 4096]} for a 2-D weight matrix.
     * Scalars are represented as an empty array {@code []}.
     */
    private final long[] shape;

    /**
     * Half-open byte range {@code [begin, end)} within the SafeTensors data
     * blob (i.e. the bytes that follow the JSON header in the file).
     * {@code end - begin == byteLength()}.
     */
    private final long dataBegin;
    private final long dataEnd;

    // ── Cached derived values ─────────────────────────────────────────────────

    /** {@code dataEnd - dataBegin}. Validated against dtype.totalBytes(shape). */
    private final long byteLength;

    /** Total number of scalar elements: product of all dimensions. */
    private final long numElements;

    // ─────────────────────────────────────────────────────────────────────────
    // Constructor (Jackson deserialization path)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Jackson-compatible constructor. Called during JSON header parsing.
     *
     * @param name        tensor name (injected after construction by the parent
     *                    header parser)
     * @param dtype       data type string from JSON (e.g. "F32")
     * @param shape       dimension array (may be empty for scalars)
     * @param dataOffsets two-element array {@code [begin, end]}
     */
    @JsonCreator
    public SafetensorTensorInfo(
            @JsonProperty("name") String name,
            @JsonProperty("dtype") String dtype,
            @JsonProperty("shape") long[] shape,
            @JsonProperty("data_offsets") long[] dataOffsets) {

        this.name = Objects.requireNonNull(name, "name must not be null");
        this.dtype = SafetensorDType.fromJson(
                Objects.requireNonNull(dtype, "dtype must not be null"));
        this.shape = shape != null ? Arrays.copyOf(shape, shape.length) : new long[0];

        if (dataOffsets == null || dataOffsets.length != 2) {
            throw new IllegalArgumentException(
                    "data_offsets must be a 2-element array [begin, end] for tensor '" + name + "'");
        }
        this.dataBegin = dataOffsets[0];
        this.dataEnd = dataOffsets[1];

        if (dataBegin < 0) {
            throw new IllegalArgumentException(
                    "data_offsets[0] must be >= 0 for tensor '" + name + "', got " + dataBegin);
        }
        if (dataEnd < dataBegin) {
            throw new IllegalArgumentException(
                    "data_offsets[1] must be >= data_offsets[0] for tensor '" + name
                            + "', got [" + dataBegin + ", " + dataEnd + "]");
        }

        this.byteLength = dataEnd - dataBegin;
        this.numElements = computeNumElements(this.shape);

        // Validate that declared byte range matches dtype × shape
        long expectedBytes = this.dtype.totalBytes(this.shape);
        if (expectedBytes >= 0 && expectedBytes != this.byteLength) {
            throw new IllegalArgumentException(
                    "Byte-length mismatch for tensor '" + name + "': "
                            + "dtype=" + dtype + " × shape=" + Arrays.toString(shape)
                            + " expects " + expectedBytes + " bytes, "
                            + "but data_offsets span " + byteLength + " bytes");
        }
    }

    // ── Builder-style static factory (non-Jackson, e.g. tests / hand-crafted) ─

    /**
     * Create a {@link SafetensorTensorInfo} directly from already-parsed values.
     *
     * @param name      tensor name
     * @param dtype     data type
     * @param shape     dimension array
     * @param dataBegin inclusive start offset in the data blob (bytes)
     * @param dataEnd   exclusive end offset in the data blob (bytes)
     * @return new instance
     */
    public static SafetensorTensorInfo of(
            String name, SafetensorDType dtype, long[] shape,
            long dataBegin, long dataEnd) {
        return new SafetensorTensorInfo(
                name, dtype.jsonName(), shape,
                new long[] { dataBegin, dataEnd });
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /** TorchTensor name as it appears in the JSON header key. */
    public String name() {
        return name;
    }

    /** Resolved element data type. */
    public SafetensorDType dtype() {
        return dtype;
    }

    /**
     * TorchTensor shape — a defensive copy is returned on every call.
     * Use {@link #rank()} + {@link #dim(int)} for hot-path access.
     */
    public long[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    /** Rank (number of dimensions). {@code 0} for scalars. */
    public int rank() {
        return shape.length;
    }

    /**
     * Size of dimension {@code i}.
     *
     * @param i zero-based dimension index
     * @return size of dimension i
     * @throws IndexOutOfBoundsException if {@code i} is out of range
     */
    public long dim(int i) {
        Objects.checkIndex(i, shape.length);
        return shape[i];
    }

    /** Inclusive byte offset within the data blob. */
    public long dataBegin() {
        return dataBegin;
    }

    /** Exclusive byte offset within the data blob. */
    public long dataEnd() {
        return dataEnd;
    }

    /** {@code dataEnd - dataBegin}. */
    public long byteLength() {
        return byteLength;
    }

    /** Total element count (product of all dimensions). 1 for scalars. */
    public long numElements() {
        return numElements;
    }

    /**
     * Whether this tensor is a scalar (zero-dimensional).
     */
    public boolean isScalar() {
        return shape.length == 0;
    }

    /**
     * Offset of element at the given multi-dimensional indices within the data
     * segment, relative to {@code dataBegin}.
     *
     * @param indices one index per dimension (C-style, row-major)
     * @return byte offset from {@code dataBegin()}
     */
    public long elementByteOffset(long... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                    "Expected " + shape.length + " indices for tensor '" + name
                            + "' but got " + indices.length);
        }
        long flatIndex = 0L;
        long stride = 1L;
        // Compute row-major flat index (last dimension fastest)
        for (int i = shape.length - 1; i >= 0; i--) {
            Objects.checkIndex(indices[i], shape[i]);
            flatIndex += indices[i] * stride;
            stride *= shape[i];
        }
        return flatIndex * dtype.byteSize();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static long computeNumElements(long[] shape) {
        if (shape.length == 0)
            return 1L; // scalar
        long n = 1L;
        for (long d : shape) {
            n *= d;
        }
        return n;
    }

    // ── Object overrides ──────────────────────────────────────────────────────

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof SafetensorTensorInfo that))
            return false;
        return dataBegin == that.dataBegin
                && dataEnd == that.dataEnd
                && Objects.equals(name, that.name)
                && dtype == that.dtype
                && Arrays.equals(shape, that.shape);
    }

    @Override
    public int hashCode() {
        int h = Objects.hash(name, dtype, dataBegin, dataEnd);
        h = 31 * h + Arrays.hashCode(shape);
        return h;
    }

    @Override
    public String toString() {
        return "SafetensorTensorInfo{"
                + "name='" + name + '\''
                + ", dtype=" + dtype.jsonName()
                + ", shape=" + Arrays.toString(shape)
                + ", data=[" + dataBegin + "," + dataEnd + ")"
                + ", bytes=" + byteLength
                + '}';
    }
}
