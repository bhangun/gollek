package tech.kayys.gollek.converter.java.gguf;

import com.google.gson.*;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * Reads HuggingFace {@code .safetensors} files using the JDK 25 FFM API.
 *
 * <p>
 * Safetensors format:
 * 
 * <pre>
 *   [8 bytes LE uint64]  – JSON header length N
 *   [N bytes UTF-8]       – JSON header  {"tensor_name": {"dtype": "F32", "shape": [..], "data_offsets": [start, end]}, "__metadata__": {...}}
 *   [data blob]           – raw tensor bytes at offsets from JSON
 * </pre>
 */
public final class SafetensorsReader implements AutoCloseable {

    public record TensorEntry(
            String name,
            String dtype, // "F32", "BF16", "F16", "I32", …
            long[] shape,
            long dataStart, // byte offset within data blob
            long dataEnd) {
        public long byteSize() {
            return dataEnd - dataStart;
        }

        public long numElements() {
            long n = 1;
            for (long d : shape)
                n *= d;
            return n;
        }

        public GgmlType ggmlType() {
            return switch (dtype) {
                case "F32" -> GgmlType.F32;
                case "F16" -> GgmlType.F16;
                case "BF16" -> GgmlType.BF16;
                case "I8" -> GgmlType.I8;
                case "I16" -> GgmlType.I16;
                case "I32" -> GgmlType.I32;
                case "I64" -> GgmlType.I64;
                case "F64" -> GgmlType.F64;
                default -> throw new UnsupportedOperationException("Unknown dtype: " + dtype);
            };
        }
    }

    private final Arena arena;
    private final MemorySegment seg;
    private final long dataOffset; // byte position where tensor data starts
    private final Map<String, TensorEntry> tensors;
    private final Map<String, String> globalMeta;

    public SafetensorsReader(Path path) throws IOException {
        arena = Arena.ofConfined();
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            seg = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size(), arena);
        }

        // Read JSON header length (8-byte LE uint64)
        long jsonLen = seg.get(
                ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 0);

        // Read JSON bytes
        byte[] jsonBytes = new byte[(int) jsonLen];
        MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, 8, jsonBytes, 0, (int) jsonLen);
        String json = new String(jsonBytes, java.nio.charset.StandardCharsets.UTF_8);

        dataOffset = 8 + jsonLen;

        // Parse header
        JsonObject root = JsonParser.parseString(json).getAsJsonObject();
        Map<String, TensorEntry> tMap = new LinkedHashMap<>();
        Map<String, String> metaMap = new LinkedHashMap<>();

        for (Map.Entry<String, JsonElement> e : root.entrySet()) {
            if (e.getKey().equals("__metadata__")) {
                JsonObject meta = e.getValue().getAsJsonObject();
                for (Map.Entry<String, JsonElement> m : meta.entrySet())
                    metaMap.put(m.getKey(), m.getValue().getAsString());
                continue;
            }
            JsonObject obj = e.getValue().getAsJsonObject();
            String dtype = obj.get("dtype").getAsString();
            JsonArray shArr = obj.get("shape").getAsJsonArray();
            long[] shape = new long[shArr.size()];
            for (int i = 0; i < shArr.size(); i++)
                shape[i] = shArr.get(i).getAsLong();
            JsonArray offsets = obj.get("data_offsets").getAsJsonArray();
            long ds = offsets.get(0).getAsLong();
            long de = offsets.get(1).getAsLong();
            tMap.put(e.getKey(), new TensorEntry(e.getKey(), dtype, shape, ds, de));
        }

        tensors = Collections.unmodifiableMap(tMap);
        globalMeta = Collections.unmodifiableMap(metaMap);
    }

    /** All tensor entries keyed by name, in declaration order. */
    public Map<String, TensorEntry> tensors() {
        return tensors;
    }

    /** Optional global metadata from {@code __metadata__} key. */
    public Map<String, String> globalMeta() {
        return globalMeta;
    }

    /**
     * Copy the raw bytes for a tensor into a freshly allocated heap array.
     * For large models prefer {@link #tensorSegment} to avoid copying.
     */
    public byte[] tensorBytes(TensorEntry entry) {
        byte[] dst = new byte[(int) entry.byteSize()];
        MemorySegment.copy(seg, ValueLayout.JAVA_BYTE,
                dataOffset + entry.dataStart(),
                dst, 0, dst.length);
        return dst;
    }

    /**
     * Return an off-heap {@link MemorySegment} slice over a tensor's data.
     * The slice shares backing memory with the mmap – no copy performed.
     * Valid until this reader is closed.
     */
    public MemorySegment tensorSegment(TensorEntry entry) {
        return seg.asSlice(dataOffset + entry.dataStart(), entry.byteSize());
    }

    @Override
    public void close() {
        arena.close();
    }
}
