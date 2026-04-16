/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorHeaderParser.java
 * ───────────────────────────
 * Low-level parser that reads the SafeTensors binary header from a
 * memory-mapped {@link MemorySegment} using the JDK 25 FFM API.
 *
 * SafeTensors binary format:
 * ┌────────────────────────────────────────────────────────────────┐
 * │  8 bytes   │ uint64, little-endian — length N of JSON header   │
 * │  N bytes   │ UTF-8 JSON string     — tensor metadata           │
 * │  variable  │ raw tensor data blob                              │
 * └────────────────────────────────────────────────────────────────┘
 *
 * Security constraints enforced here:
 *  - Header JSON length is capped at {@value #MAX_HEADER_BYTES} bytes to
 *    prevent integer overflow and heap exhaustion on corrupt files.
 *  - All offset arithmetic is performed in {@code long} to avoid int overflow
 *    on large (> 2 GiB) model files.
 *  - Header JSON is validated to ensure data_offsets are monotonic and
 *    non-overlapping before the caller gets a {@link SafetensorHeader}.
 */
package tech.kayys.gollek.safetensor.loader;

import com.fasterxml.jackson.databind.ObjectMapper;

import tech.kayys.gollek.safetensor.exception.SafetensorException;

import org.jboss.logging.Logger;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;

/**
 * Stateless, thread-safe SafeTensors header parser backed by FFM API.
 *
 * <p>
 * Instances should be obtained via the Quarkus CDI container or by calling
 * {@link #create(ObjectMapper)}. The {@link ObjectMapper} is expected to be
 * the application-wide Quarkus-managed instance.
 */
public final class SafetensorHeaderParser {

    private static final Logger log = Logger.getLogger(SafetensorHeaderParser.class);

    /**
     * Maximum allowed header JSON length (100 MiB).
     * Any file claiming a header larger than this is almost certainly corrupt.
     */
    static final long MAX_HEADER_BYTES = 100L * 1024 * 1024; // 100 MiB

    /**
     * Minimum valid file size: 8-byte length prefix + 2-byte minimal JSON ("{}").
     */
    private static final long MIN_FILE_BYTES = 10L;

    /** Little-endian uint64 layout for reading the 8-byte header length. */
    private static final ValueLayout.OfLong UINT64_LE = ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final ObjectMapper mapper;

    // ─────────────────────────────────────────────────────────────────────────

    private SafetensorHeaderParser(ObjectMapper mapper) {
        this.mapper = Objects.requireNonNull(mapper, "mapper must not be null");
    }

    /**
     * Create a new parser backed by the provided {@link ObjectMapper}.
     *
     * @param mapper Jackson mapper (must support standard types; no special config
     *               needed)
     * @return a new, stateless parser instance
     */
    public static SafetensorHeaderParser create(ObjectMapper mapper) {
        return new SafetensorHeaderParser(mapper);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public parse API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Parse the SafeTensors header from a memory-mapped file segment.
     *
     * <p>
     * The {@code fileSegment} must cover the entire file (or at least the
     * header region — up to {@code 8 + headerLength} bytes). The parser does
     * NOT close the segment; lifetime is managed by the caller's Arena.
     *
     * @param fileSegment memory segment of the raw file bytes (may be mmap'd)
     * @param filePath    path to the source file, used only in error messages
     * @return the decoded {@link SafetensorHeader}
     * @throws SafetensorException.HeaderParseException on malformed JSON or
     *                                                  truncated data
     * @throws SafetensorException.ValidationException  on semantic violations
     * @throws SafetensorException.IoException          on segment access failures
     */
    public SafetensorHeader parse(MemorySegment fileSegment, Path filePath) {
        long fileSize = fileSegment.byteSize();
        if (fileSize < MIN_FILE_BYTES) {
            throw new SafetensorException.ValidationException(
                    "File too small to be a valid SafeTensors file "
                            + "(size=" + fileSize + ", minimum=" + MIN_FILE_BYTES + ")",
                    filePath);
        }

        // ── Step 1: read 8-byte little-endian header length ───────────────────
        long headerJsonLength = readUint64LE(fileSegment, 0L, filePath);
        log.debugf("SafeTensors header JSON length: %d bytes [%s]", headerJsonLength, filePath);

        // Sanity checks on the length value
        if (headerJsonLength <= 0) {
            throw new SafetensorException.ValidationException(
                    "Header JSON length must be > 0, got " + headerJsonLength, filePath);
        }
        if (headerJsonLength > MAX_HEADER_BYTES) {
            throw new SafetensorException.ValidationException(
                    "Header JSON length " + headerJsonLength
                            + " bytes exceeds safety cap of " + MAX_HEADER_BYTES + " bytes. "
                            + "File may be corrupt.",
                    filePath);
        }

        long dataBlobOffset = 8L + headerJsonLength; // absolute offset of tensor data

        if (dataBlobOffset > fileSize) {
            throw new SafetensorException.ValidationException(
                    "Header claims JSON length=" + headerJsonLength
                            + " but file size=" + fileSize
                            + " — file is truncated.",
                    filePath);
        }

        // ── Step 2: extract JSON bytes as a sub-segment (zero-copy) ──────────
        MemorySegment jsonSegment = fileSegment.asSlice(8L, headerJsonLength);
        String jsonString = readUtf8(jsonSegment, filePath);

        // ── Step 3: parse JSON into tensor info map ───────────────────────────
        Map<String, SafetensorTensorInfo> tensors = new LinkedHashMap<>();
        Map<String, String> fileMeta = new LinkedHashMap<>();
        parseJson(jsonString, tensors, fileMeta, filePath);

        // ── Step 4: semantic validation ───────────────────────────────────────
        validateTensorOffsets(tensors, fileSize, dataBlobOffset, filePath);

        SafetensorHeader header = SafetensorHeader.of(dataBlobOffset, tensors, fileMeta);

        log.infof("Parsed SafeTensors header: %d tensors, dataBlob@%d [%s]",
                header.tensorCount(), header.dataBlobOffset(), filePath);

        return header;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Read a little-endian uint64 from the given byte offset in the segment.
     * Java's {@code long} is signed; we treat the bits as unsigned by checking
     * for negative values (which indicate files > 2^63 — practically impossible).
     */
    private static long readUint64LE(MemorySegment seg, long offset, Path path) {
        try {
            long value = seg.get(UINT64_LE, offset);
            if (value < 0) {
                throw new SafetensorException.ValidationException(
                        "Header JSON length exceeds Long.MAX_VALUE — file is corrupt.", path);
            }
            return value;
        } catch (IndexOutOfBoundsException e) {
            throw new SafetensorException.HeaderParseException(
                    "Cannot read 8-byte length prefix (file too small?)", path, 0L, e);
        }
    }

    /**
     * Decode the JSON bytes from the memory segment as UTF-8 without heap copy.
     *
     * <p>
     * The {@link MemorySegment#getString} call uses a temporary heap copy
     * internally; for headers up to ~100 MiB this is acceptable. For very large
     * headers a streaming JSON parser could be wired here instead.
     */
    private static String readUtf8(MemorySegment jsonSeg, Path path) {
        try {
            // MemorySegment.getString reads UTF-8 null-terminated; we use the
            // array path to get the exact slice
            byte[] bytes = jsonSeg.toArray(ValueLayout.JAVA_BYTE);
            return new String(bytes, StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new SafetensorException.HeaderParseException(
                    "Failed to decode JSON header as UTF-8", path, 8L, e);
        }
    }

    /**
     * Parse the JSON string into the output maps, handling the mixed-type header:
     * {@code __metadata__} → string map, everything else →
     * {@link SafetensorTensorInfo}.
     */
    @SuppressWarnings("unchecked")
    private void parseJson(
            String jsonString,
            Map<String, SafetensorTensorInfo> tensors,
            Map<String, String> fileMeta,
            Path path) {
        try {
            Map<String, Object> root = mapper.readValue(jsonString, LinkedHashMap.class);

            for (Map.Entry<String, Object> entry : root.entrySet()) {
                String key = entry.getKey();
                Object value = entry.getValue();

                if ("__metadata__".equals(key)) {
                    if (value instanceof Map<?, ?> metaMap) {
                        metaMap.forEach(
                                (k, v) -> fileMeta.put(String.valueOf(k), v != null ? String.valueOf(v) : null));
                    }
                } else {
                    if (!(value instanceof Map<?, ?> tensorMap)) {
                        throw new SafetensorException.HeaderParseException(
                                "Expected an object for tensor key '" + key
                                        + "', got " + (value == null ? "null" : value.getClass().getSimpleName()),
                                path, -1L, null);
                    }
                    SafetensorTensorInfo info = buildTensorInfo(key, (Map<String, Object>) tensorMap, path);
                    tensors.put(key, info);
                }
            }
        } catch (SafetensorException e) {
            throw e;
        } catch (Exception e) {
            throw new SafetensorException.HeaderParseException(
                    "JSON parsing failed: " + e.getMessage(), path, -1L, e);
        }
    }

    @SuppressWarnings("unchecked")
    private static SafetensorTensorInfo buildTensorInfo(
            String name, Map<String, Object> raw, Path path) {
        try {
            Object dtypeObj = raw.get("dtype");
            if (!(dtypeObj instanceof String dtype)) {
                throw new SafetensorException.HeaderParseException(
                        "Missing or non-string 'dtype' for tensor '" + name + "'", path, -1L, null);
            }

            List<Number> shapeList = (List<Number>) raw.get("shape");
            long[] shape = shapeList == null
                    ? new long[0]
                    : shapeList.stream().mapToLong(Number::longValue).toArray();

            List<Number> offsetList = (List<Number>) raw.get("data_offsets");
            if (offsetList == null || offsetList.size() != 2) {
                throw new SafetensorException.HeaderParseException(
                        "data_offsets must be a 2-element array for tensor '" + name + "'",
                        path, -1L, null);
            }
            long begin = offsetList.get(0).longValue();
            long end = offsetList.get(1).longValue();

            return new SafetensorTensorInfo(name, dtype, shape, new long[] { begin, end });

        } catch (SafetensorException e) {
            throw e;
        } catch (IllegalArgumentException e) {
            throw new SafetensorException.HeaderParseException(
                    "Invalid tensor definition for '" + name + "': " + e.getMessage(),
                    path, -1L, e);
        }
    }

    /**
     * Validate that all tensor data_offsets are within bounds and non-overlapping.
     *
     * <p>
     * Non-overlapping is not strictly required by the SafeTensors spec but is
     * a strong indicator of file integrity and protects against pathological
     * alias scenarios.
     */
    private static void validateTensorOffsets(
            Map<String, SafetensorTensorInfo> tensors,
            long fileSize,
            long dataBlobOffset,
            Path path) {

        // Collect and sort all intervals by begin offset
        List<SafetensorTensorInfo> sorted = new ArrayList<>(tensors.values());
        sorted.sort(Comparator.comparingLong(SafetensorTensorInfo::dataBegin));

        long prevEnd = 0L;

        for (SafetensorTensorInfo info : sorted) {
            long absBegin = dataBlobOffset + info.dataBegin();
            long absEnd = dataBlobOffset + info.dataEnd();

            if (absEnd > fileSize) {
                throw new SafetensorException.ValidationException(
                        "TorchTensor '" + info.name() + "' data_offsets [" + info.dataBegin()
                                + "," + info.dataEnd() + ") extend beyond file size " + fileSize
                                + " (absolute range [" + absBegin + "," + absEnd + "))",
                        path, info.name());
            }

            // Allow zero-length tensors at the same position (degenerate but valid)
            if (info.byteLength() > 0 && info.dataBegin() < prevEnd) {
                throw new SafetensorException.ValidationException(
                        "TorchTensor '" + info.name() + "' overlaps with a previous tensor. "
                                + "Begin offset " + info.dataBegin() + " < previous end " + prevEnd,
                        path, info.name());
            }
            prevEnd = Math.max(prevEnd, info.dataEnd());
        }
    }
}
