/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorHeader.java
 * ─────────────────────
 * Represents the decoded SafeTensors JSON header — the first logical section
 * of every .safetensors file.
 *
 * Binary layout of a SafeTensors file
 * ════════════════════════════════════
 *  Byte 0–7   : uint64 little-endian  — length of the JSON header (N bytes)
 *  Byte 8–8+N : UTF-8 JSON blob       — the header object
 *  Byte 8+N.. : raw tensor data blob  — referenced by data_offsets in header
 *
 * JSON header structure
 * ═════════════════════
 *  {
 *    "__metadata__": { "format": "pt", "description": "...", ... },
 *    "model.embed_tokens.weight": {
 *        "dtype": "BF16",
 *        "shape": [32000, 4096],
 *        "data_offsets": [0, 262144000]
 *    },
 *    ...
 *  }
 *
 * This class holds:
 *  - a lookup map for O(1) tensor access by name
 *  - the raw __metadata__ string map (optional)
 *  - the absolute data-blob start offset within the file (= 8 + headerLength)
 *
 * Thread-safety: immutable after construction.
 */
package tech.kayys.gollek.safetensor.loader;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;

import java.io.IOException;
import java.util.*;

/**
 * Decoded SafeTensors JSON header.
 *
 * <p>
 * This is the full, structured view of everything between the 8-byte length
 * prefix and the first tensor byte. It is produced by
 * {@link tech.kayys.gollek.inference.safetensor.ffm.SafetensorHeaderParser}
 * and consumed by the tensor accessor layer.
 */
@JsonDeserialize(using = SafetensorHeader.Deserializer.class)
public final class SafetensorHeader {

    // ── Reserved JSON key for file-level metadata ─────────────────────────────
    private static final String METADATA_KEY = "__metadata__";

    /** Absolute file offset of byte 0 of the data blob (= 8 + headerJsonLength). */
    private final long dataBlobOffset;

    /** Per-tensor metadata, keyed by tensor name. Insertion-ordered. */
    private final Map<String, SafetensorTensorInfo> tensors;

    /** Optional __metadata__ string map (provenance, version, etc.). */
    private final Map<String, String> fileMetadata;

    // ─────────────────────────────────────────────────────────────────────────

    private SafetensorHeader(
            long dataBlobOffset,
            Map<String, SafetensorTensorInfo> tensors,
            Map<String, String> fileMetadata) {
        this.dataBlobOffset = dataBlobOffset;
        this.tensors = Collections.unmodifiableMap(new LinkedHashMap<>(tensors));
        this.fileMetadata = fileMetadata != null
                ? Collections.unmodifiableMap(new LinkedHashMap<>(fileMetadata))
                : Collections.emptyMap();
    }

    // ── Factory ───────────────────────────────────────────────────────────────

    /**
     * Build a header from already-parsed components. Used by
     * {@link tech.kayys.gollek.inference.safetensor.ffm.SafetensorHeaderParser}.
     *
     * @param dataBlobOffset absolute file byte offset where tensor data begins
     * @param tensors        per-tensor info map (will be defensively copied)
     * @param fileMetadata   __metadata__ string map (may be null / empty)
     */
    public static SafetensorHeader of(
            long dataBlobOffset,
            Map<String, SafetensorTensorInfo> tensors,
            Map<String, String> fileMetadata) {
        Objects.requireNonNull(tensors, "tensors must not be null");
        if (dataBlobOffset < 8) {
            throw new IllegalArgumentException(
                    "dataBlobOffset must be >= 8 (8-byte length prefix + at least one JSON byte), got "
                            + dataBlobOffset);
        }
        return new SafetensorHeader(dataBlobOffset, tensors, fileMetadata);
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /**
     * Absolute file-byte offset at which the raw tensor data starts.
     * All {@code data_offsets} inside {@link SafetensorTensorInfo} are relative
     * to this offset.
     */
    public long dataBlobOffset() {
        return dataBlobOffset;
    }

    /** Number of tensors stored in this file (or shard). */
    public int tensorCount() {
        return tensors.size();
    }

    /** Unmodifiable ordered map of tensor name → metadata. */
    public Map<String, SafetensorTensorInfo> tensors() {
        return tensors;
    }

    /**
     * Look up tensor info by name.
     *
     * @param name tensor key (e.g.
     *             {@code "model.layers.0.self_attn.q_proj.weight"})
     * @return the tensor descriptor
     * @throws java.util.NoSuchElementException if the tensor is not in this header
     */
    public SafetensorTensorInfo tensor(String name) {
        SafetensorTensorInfo info = tensors.get(name);
        if (info == null) {
            throw new NoSuchElementException(
                    "No tensor named '" + name + "' in this SafeTensors header. "
                            + "Available tensors: " + tensors.keySet());
        }
        return info;
    }

    /**
     * Look up tensor info, returning {@link Optional#empty()} if absent.
     * Preferred for cases where the tensor may or may not be present
     * (e.g. optional bias tensors).
     */
    public Optional<SafetensorTensorInfo> findTensor(String name) {
        return Optional.ofNullable(tensors.get(name));
    }

    /** {@code true} if a tensor with the given name exists in this header. */
    public boolean hasTensor(String name) {
        return tensors.containsKey(name);
    }

    /**
     * Optional file-level metadata (the {@code __metadata__} JSON object).
     * Keys and values are raw strings from the JSON.
     * Returns an empty map if the file has no {@code __metadata__} block.
     */
    public Map<String, String> fileMetadata() {
        return fileMetadata;
    }

    /**
     * Get a specific file-metadata entry.
     *
     * @param key metadata key
     * @return the value, or {@link Optional#empty()} if absent
     */
    public Optional<String> metadataValue(String key) {
        return Optional.ofNullable(fileMetadata.get(key));
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /**
     * Total raw data bytes across all tensors in this header.
     * Useful for pre-allocating a contiguous buffer or reporting model size.
     */
    public long totalDataBytes() {
        return tensors.values().stream()
                .mapToLong(SafetensorTensorInfo::byteLength)
                .sum();
    }

    /**
     * Absolute file-byte offset of the first byte of {@code tensorName}'s data.
     *
     * @param tensorName tensor key
     * @return absolute byte offset in the file
     */
    public long absoluteDataBegin(String tensorName) {
        return dataBlobOffset + tensor(tensorName).dataBegin();
    }

    /**
     * Absolute file-byte offset one past the last byte of {@code tensorName}'s
     * data.
     */
    public long absoluteDataEnd(String tensorName) {
        return dataBlobOffset + tensor(tensorName).dataEnd();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Custom Jackson Deserializer
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Custom deserialiser that handles the mixed-type JSON header:
     * <ul>
     * <li>{@code "__metadata__"} → {@code Map<String,String>}
     * <li>All other keys → {@link SafetensorTensorInfo} objects
     * </ul>
     *
     * <p>
     * The {@code dataBlobOffset} is injected after deserialization by
     * {@link tech.kayys.gollek.inference.safetensor.ffm.SafetensorHeaderParser}
     * because it depends on the byte-level header length, which is known only
     * during file parsing.
     */
    static final class Deserializer extends StdDeserializer<SafetensorHeader> {

        Deserializer() {
            super(SafetensorHeader.class);
        }

        @Override
        public SafetensorHeader deserialize(JsonParser p, DeserializationContext ctx)
                throws IOException {

            Map<String, SafetensorTensorInfo> tensors = new LinkedHashMap<>();
            Map<String, String> fileMeta = new LinkedHashMap<>();

            p.nextToken(); // move past START_OBJECT

            while (p.currentToken() == JsonToken.FIELD_NAME) {
                String fieldName = p.currentName();
                p.nextToken(); // move to value

                if (METADATA_KEY.equals(fieldName)) {
                    // __metadata__ is a flat string → string map
                    fileMeta = ctx.readValue(p, ctx.getTypeFactory()
                            .constructMapType(LinkedHashMap.class, String.class, String.class));
                } else {
                    // All other keys are tensor descriptors
                    // We temporarily read as a raw map, then construct TensorInfo
                    @SuppressWarnings("unchecked")
                    Map<String, Object> raw = (Map<String, Object>) ctx.readValue(p, Map.class);

                    SafetensorTensorInfo info = buildTensorInfo(fieldName, raw);
                    tensors.put(fieldName, info);
                }

                p.nextToken(); // advance to next FIELD_NAME or END_OBJECT
            }

            // dataBlobOffset is unknown at JSON-parse time; use sentinel 8
            // The loader will reconstruct with the real offset after it knows headerLen.
            return new SafetensorHeader(8L, tensors, fileMeta);
        }

        @SuppressWarnings("unchecked")
        private static SafetensorTensorInfo buildTensorInfo(String name, Map<String, Object> raw) {
            String dtype = (String) raw.get("dtype");

            List<Number> shapeList = (List<Number>) raw.get("shape");
            long[] shape = shapeList == null
                    ? new long[0]
                    : shapeList.stream().mapToLong(Number::longValue).toArray();

            List<Number> offsetList = (List<Number>) raw.get("data_offsets");
            if (offsetList == null || offsetList.size() != 2) {
                throw new IllegalStateException(
                        "data_offsets must have 2 elements for tensor '" + name + "'");
            }
            long begin = offsetList.get(0).longValue();
            long end = offsetList.get(1).longValue();

            return new SafetensorTensorInfo(name, dtype, shape, new long[] { begin, end });
        }
    }

    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public String toString() {
        return "SafetensorHeader{tensors=" + tensors.size()
                + ", dataBlobOffset=" + dataBlobOffset
                + ", totalBytes=" + totalDataBytes()
                + '}';
    }
}
