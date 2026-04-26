package tech.kayys.gollek.converter.java.gguf;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads a GGUF v3 file into a {@link GgufModel} using the
 * JDK 25 Foreign Function &amp; Memory (FFM) API.
 *
 * <p>
 * The entire file is memory-mapped via {@link FileChannel#map} through
 * {@link Arena} so no heap copies are needed for tensor data.
 */
public final class GgufReader implements AutoCloseable {

    private final Arena arena;
    private final MemorySegment seg; // entire file mapped read-only
    private long pos; // current read cursor (byte offset)
    private long metadataEnd = 0; // end position of metadata section

    // ── Construction ──────────────────────────────────────────────────────

    public GgufReader(Path path) throws IOException {
        arena = Arena.ofConfined();
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            seg = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size(), arena);
        }
        pos = 0;
    }

    // ── Public entry point ────────────────────────────────────────────────

    /** Parse the mapped file and return the populated {@link GgufModel}. */
    public GgufModel read() {
        GgufModel model = new GgufModel();

        // ── Header (24 bytes) ─────────────────────────────────────────
        int magic = readI32LE();
        if (magic != GgufModel.MAGIC)
            throw new IllegalStateException("Not a GGUF file (bad magic)");

        int version = readI32LE();
        if (version < 1 || version > 3)
            throw new IllegalStateException("Unsupported GGUF version: " + version);

        long nTensors = readI64LE();
        long nKv = readI64LE();

        if (nKv > 1_000_000)
            throw new IllegalStateException("Implausibly large KV count: " + nKv);
        if (nTensors > 1_000_000)
            throw new IllegalStateException("Implausibly large tensor count: " + nTensors);

        // ── Metadata key-value pairs ──────────────────────────────────
        for (long i = 0; i < nKv; i++) {
            String key = readString();
            GgufMetaType vtype = GgufMetaType.fromId(readI32LE());
            GgufMetaValue val = readValue(vtype);
            model.addMeta(key, val);
        }

        // ── Tensor descriptors ────────────────────────────────────────
        List<TensorInfo> infos = new ArrayList<>((int) nTensors);
        for (long i = 0; i < nTensors; i++) {
            String name = readString();
            int nDims = readI32LE();
            long[] ne = new long[nDims];
            for (int d = 0; d < nDims; d++)
                ne[d] = readI64LE();
            GgmlType type = GgmlType.fromId(readI32LE());
            long off = readI64LE();
            infos.add(new TensorInfo(name, ne, type, off));
        }

        // Store metadata end position before alignment padding
        metadataEnd = pos;

        // ── Alignment padding before tensor data ──────────────────────
        int alignment = model.alignment();
        long dataStart = alignUp(pos, alignment);

        // ── Register tensor descriptors with corrected offsets ────────
        for (TensorInfo info : infos) {
            model.addTensor(info); // offsets are relative to dataStart already
        }

        // ── Copy tensor data blob ─────────────────────────────────────
        long dataLen = seg.byteSize() - dataStart;
        if (dataLen > 0) {
            byte[] blob = new byte[(int) dataLen];
            MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, dataStart,
                    blob, 0, (int) dataLen);
            model.setTensorData(blob);
        }

        return model;
    }

    /**
     * Read entire GGUF file into memory (for small models and validation).
     * For files > 2 GB, this will throw an exception with guidance.
     */
    public GgufModel readIntoModel() throws IOException {
        GgufModel model = read();

        // If tensor data is not already loaded, load it for validation
        if (model.tensorData() == null && !model.tensors().isEmpty()) {
            long dataStart = alignUp(metadataEnd, model.alignment());
            long dataLen = seg.byteSize() - dataStart;

            if (dataLen > Integer.MAX_VALUE) {
                throw new IllegalStateException(
                    "Model tensor data section (" + dataLen + " bytes) exceeds 2 GB limit. " +
                    "Use tensorData(TensorInfo) for zero-copy access instead."
                );
            }

            byte[] data = new byte[(int) dataLen];
            MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, dataStart, data, 0, (int) dataLen);
            model.setTensorData(data);
        }

        return model;
    }

    // ── Close (release arena / mapped memory) ─────────────────────────────

    @Override
    public void close() {
        arena.close();
    }

    // ── Low-level read helpers ────────────────────────────────────────────

    private int readI32LE() {
        int v = seg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos);
        pos += 4;
        return v;
    }

    private long readI64LE() {
        long v = seg.get(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos);
        pos += 8;
        return v;
    }

    private float readF32LE() {
        float v = seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos);
        pos += 4;
        return v;
    }

    private double readF64LE() {
        double v = seg.get(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos);
        pos += 8;
        return v;
    }

    private short readI16LE() {
        short v = seg.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos);
        pos += 2;
        return v;
    }

    private byte readByte() {
        byte v = seg.get(ValueLayout.JAVA_BYTE, pos);
        pos += 1;
        return v;
    }

    /**
     * GGUF string: uint64 length prefix + UTF-8 bytes (NOT null-terminated).
     */
    private String readString() {
        long len = readI64LE();
        if (len == 0)
            return "";
        byte[] bytes = new byte[(int) len];
        MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, pos, bytes, 0, (int) len);
        pos += len;
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
    }

    private GgufMetaValue readValue(GgufMetaType type) {
        return switch (type) {
            case UINT8 -> new GgufMetaValue.UInt8Val((short) (readByte() & 0xFF));
            case INT8 -> new GgufMetaValue.Int8Val(readByte());
            case UINT16 -> new GgufMetaValue.UInt16Val(readI16LE() & 0xFFFF);
            case INT16 -> new GgufMetaValue.Int16Val(readI16LE());
            case UINT32 -> new GgufMetaValue.UInt32Val(Integer.toUnsignedLong(readI32LE()));
            case INT32 -> new GgufMetaValue.Int32Val(readI32LE());
            case FLOAT32 -> new GgufMetaValue.Float32Val(readF32LE());
            case BOOL -> new GgufMetaValue.BoolVal(readByte() != 0);
            case STRING -> new GgufMetaValue.StringVal(readString());
            case UINT64 -> new GgufMetaValue.UInt64Val(readI64LE());
            case INT64 -> new GgufMetaValue.Int64Val(readI64LE());
            case FLOAT64 -> new GgufMetaValue.Float64Val(readF64LE());
            case ARRAY -> {
                GgufMetaType elemType = GgufMetaType.fromId(readI32LE());
                long count = readI64LE();
                List<GgufMetaValue> elems = new ArrayList<>((int) count);
                for (long i = 0; i < count; i++)
                    elems.add(readValue(elemType));
                yield new GgufMetaValue.ArrayVal(elemType, elems);
            }
        };
    }

    private static long alignUp(long offset, long alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }
}
