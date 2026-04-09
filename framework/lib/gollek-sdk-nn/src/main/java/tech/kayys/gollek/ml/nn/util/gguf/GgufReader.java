package tech.kayys.gollek.ml.nn.util.gguf;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * SDK-friendly GGUF Reader.
 * Supports dequantization of Q4_0 and Q8_0 tensors.
 */
public final class GgufReader implements AutoCloseable {

    private static final int MAGIC = 0x46554747; // "GGUF" in LE

    private final Arena arena;
    private final MemorySegment seg;
    private long pos;
    private final Map<String, GgufMetaValue> metadata = new LinkedHashMap<>();
    private long nTensors;

    public GgufReader(Path path) throws IOException {
        this.arena = Arena.ofConfined();
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            this.seg = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size(), arena);
        }
        this.pos = 0;
        readHeaderAndMetadata();
    }

    private void readHeaderAndMetadata() {
        int magic = readI32();
        if (magic != MAGIC) throw new IllegalStateException("Not a GGUF file");
        int version = readI32();
        if (version < 2) throw new IllegalStateException("Unsupported GGUF version: " + version);

        this.nTensors = readI64();
        long nKv = readI64();

        for (long i = 0; i < nKv; i++) {
            String key = readString();
            GgufMetaType type = GgufMetaType.fromId(readI32());
            metadata.put(key, readValue(type));
        }
    }

    public Map<String, GgufMetaValue> getMetadata() {
        return Collections.unmodifiableMap(metadata);
    }

    public Map<String, float[]> loadTensors() {
        // 3. Tensor infos
        List<TensorDesc> descs = new ArrayList<>();
        for (long i = 0; i < nTensors; i++) {
            String name = readString();
            int nDims = readI32();
            long[] shape = new long[nDims];
            long numel = 1;
            for (int d = 0; d < nDims; d++) {
                shape[d] = readI64();
                numel *= shape[d];
            }
            GgmlType type = GgmlType.fromId(readI32());
            long offset = readI64();
            descs.add(new TensorDesc(name, shape, numel, type, offset));
        }

        // 4. Alignment
        long alignment = 32;
        if (metadata.containsKey("general.alignment")) {
            alignment = ((GgufMetaValue.Uint32Val) metadata.get("general.alignment")).value();
        }

        long dataStart = (pos + alignment - 1) & ~(alignment - 1);

        // 5. Load/Dequantize
        Map<String, float[]> results = new LinkedHashMap<>();
        for (TensorDesc desc : descs) {
            long absoluteOffset = dataStart + desc.offset;
            float[] data = dequantize(desc, absoluteOffset);
            results.put(desc.name, data);
        }

        return results;
    }

    private float[] dequantize(TensorDesc desc, long offset) {
        float[] result = new float[(int) desc.numel];
        GgmlType type = desc.type;

        switch (type) {
            case F32 -> {
                for (int i = 0; i < desc.numel; i++) {
                    result[i] = seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), offset + i * 4L);
                }
            }
            case F16 -> {
                for (int i = 0; i < desc.numel; i++) {
                    short h = seg.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), offset + i * 2L);
                    result[i] = Float.float16ToFloat(h);
                }
            }
            case Q8_0 -> {
                // block: 2 bytes float16 scale + 32 bytes int8 weights
                int blocks = (int) (desc.numel / 32);
                for (int b = 0; b < blocks; b++) {
                    long blockOff = offset + b * 34L;
                    short hScale = seg.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), blockOff);
                    float scale = Float.float16ToFloat(hScale);
                    for (int i = 0; i < 32; i++) {
                        byte w = seg.get(ValueLayout.JAVA_BYTE, blockOff + 2 + i);
                        result[b * 32 + i] = w * scale;
                    }
                }
            }
            case Q4_0 -> {
                // block: 2 bytes float16 scale + 16 bytes (32 * 4bit weights)
                int blocks = (int) (desc.numel / 32);
                for (int b = 0; b < blocks; b++) {
                    long blockOff = offset + b * 18L;
                    short hScale = seg.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), blockOff);
                    float scale = Float.float16ToFloat(hScale);
                    for (int i = 0; i < 16; i++) {
                        byte pack = seg.get(ValueLayout.JAVA_BYTE, blockOff + 2 + i);
                        byte w0 = (byte) (pack & 0x0F);
                        byte w1 = (byte) ((pack >> 4) & 0x0F);
                        // Map 0..15 to -8..7
                        result[b * 32 + i] = (w0 - 8) * scale;
                        result[b * 32 + i + 16] = (w1 - 8) * scale;
                    }
                }
            }
            default -> throw new UnsupportedOperationException("SDK dequantization not implemented for: " + type);
        }
        return result;
    }

    private int readI32() {
        int v = seg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos);
        pos += 4;
        return v;
    }

    private long readI64() {
        long v = seg.get(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos);
        pos += 8;
        return v;
    }

    private String readString() {
        long len = readI64();
        byte[] bytes = new byte[(int) len];
        MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, pos, bytes, 0, (int) len);
        pos += len;
        return new String(bytes);
    }

    private GgufMetaValue readValue(GgufMetaType type) {
        return switch (type) {
            case UINT8 -> new GgufMetaValue.Uint8Val((short) (seg.get(ValueLayout.JAVA_BYTE, pos++) & 0xFF));
            case INT8 -> new GgufMetaValue.Int8Val(seg.get(ValueLayout.JAVA_BYTE, pos++));
            case UINT32 -> new GgufMetaValue.Uint32Val(Integer.toUnsignedLong(readI32()));
            case INT32 -> new GgufMetaValue.Int32Val(readI32());
            case FLOAT32 -> new GgufMetaValue.Float32Val(seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), pos += 4)); // simplified
            case STRING -> new GgufMetaValue.StringVal(readString());
            case ARRAY -> {
                GgufMetaType eType = GgufMetaType.fromId(readI32());
                long count = readI64();
                List<GgufMetaValue> list = new ArrayList<>();
                for (long i = 0; i < count; i++) list.add(readValue(eType));
                yield new GgufMetaValue.ArrayVal(eType, list);
            }
            default -> {
                // Skip unknown types
                pos += typeSize(type);
                yield null;
            }
        };
    }

    private int typeSize(GgufMetaType type) {
        return switch (type) {
            case UINT8, INT8, BOOL -> 1;
            case UINT16, INT16 -> 2;
            case UINT32, INT32, FLOAT32 -> 4;
            case UINT64, INT64, FLOAT64 -> 8;
            default -> 0;
        };
    }

    @Override
    public void close() {
        arena.close();
    }

    private record TensorDesc(String name, long[] shape, long numel, GgmlType type, long offset) {}
}
