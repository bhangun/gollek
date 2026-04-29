package tech.kayys.gollek.gguf.loader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure Java GGUF parser. Extracts model structure and metadata.
 */
public final class GGUFParser {

    public GGUFModel parse(MemorySegment seg, Arena arena) {
        GGUFReader.Cursor c = new GGUFReader.Cursor(seg);

        // 1. Header
        byte[] magic = new byte[4];
        for (int i = 0; i < 4; i++) magic[i] = c.i8();
        if (!new String(magic).equals("GGUF")) {
            throw new IllegalArgumentException("Not a valid GGUF file");
        }

        int version = c.i32();
        long nTensors = c.i64();
        long nKv = c.i64();

        // 2. Metadata KV
        Map<String, Object> metadata = new HashMap<>();
        for (int i = 0; i < nKv; i++) {
            String key = c.str();
            int typeId = c.i32();
            metadata.put(key, readValue(c, typeId));
        }

        // 3. Tensor Info
        List<GGUFTensorInfo> tensors = new ArrayList<>();
        for (int i = 0; i < nTensors; i++) {
            String name = c.str();
            int nDims = c.i32();
            long[] shape = new long[nDims];
            long numElements = 1;
            for (int d = 0; d < nDims; d++) {
                shape[d] = c.i64();
                numElements *= shape[d];
            }
            int typeId = c.i32();
            long offset = c.i64();
            long size = calculateTensorSize(numElements, typeId);
            GGUFTensorInfo info = new GGUFTensorInfo(name, shape, typeId, offset, size);
            tensors.add(info);
        }

        // Alignment - GGUF usually aligns data to 32 bytes by default
        int alignment = ((Number) metadata.getOrDefault("general.alignment", 32)).intValue();
        long pos = c.position();
        long padding = (alignment - (pos % alignment)) % alignment;
        long dataStart = pos + padding;

        return new GGUFModel(version, metadata, tensors, dataStart, seg, arena);
    }

    private Object readValue(GGUFReader.Cursor c, int typeId) {
        return switch (typeId) {
            case 0 -> (int) c.i8() & 0xFF; // UINT8
            case 1 -> (int) c.i8();        // INT8
            case 2 -> c.u16();             // UINT16
            case 3 -> (int) c.i16();       // INT16
            case 4 -> c.u32();             // UINT32
            case 5 -> c.i32();             // INT32
            case 6 -> c.f32();             // FLOAT32
            case 7 -> c.i8() != 0;         // BOOL
            case 8 -> c.str();             // STRING
            case 9 -> readArray(c);        // ARRAY
            case 10 -> c.i64();            // UINT64
            case 11 -> c.i64();            // INT64
            case 12 -> c.f64();            // FLOAT64
            default -> throw new UnsupportedOperationException("Unsupported GGUF KV type: " + typeId + " at pos " + c.position());
        };
    }

    private List<Object> readArray(GGUFReader.Cursor c) {
        int typeId = c.i32();
        long len = c.i64();
        List<Object> list = new ArrayList<>((int) len);
        for (int i = 0; i < len; i++) {
            list.add(readValue(c, typeId));
        }
        return list;
    }
    
    private long calculateTensorSize(long numElements, int typeId) {
        return switch (typeId) {
            case 0 -> numElements * 4; // F32
            case 1 -> numElements * 2; // F16
            case 2 -> (numElements / 32) * 18; // Q4_0
            case 3 -> (numElements / 32) * 20; // Q4_1
            case 6 -> (numElements / 32) * 18; // Q5_0
            case 7 -> (numElements / 32) * 22; // Q5_1
            case 8 -> (numElements / 32) * 34; // Q8_0
            case 9 -> numElements * 2; // Q8_1
            case 10 -> (numElements / 256) * 84; // Q2_K
            case 11 -> (numElements / 256) * 110; // Q3_K
            case 12 -> (numElements / 256) * 144; // Q4_K
            case 13 -> (numElements / 256) * 176; // Q5_K
            case 14 -> (numElements / 256) * 210; // Q6_K
            case 15 -> (numElements / 256) * 344; // Q8_K
            case 16 -> (numElements / 256) * 73; // IQ2_XXS
            case 17 -> (numElements / 256) * 80; // IQ2_XS
            case 18 -> (numElements / 256) * 66; // IQ1_S
            case 19 -> (numElements / 256) * 76; // IQ1_M
            case 20 -> (numElements / 256) * 89; // IQ2_S
            case 21 -> (numElements / 256) * 105; // IQ3_XXS
            case 22 -> (numElements / 256) * 114; // IQ3_S
            default -> numElements * 4; // Fallback
        };
    }
}
