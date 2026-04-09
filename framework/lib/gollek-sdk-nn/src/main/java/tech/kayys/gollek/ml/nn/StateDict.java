package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Save and load model parameters in SafeTensors format using JDK 25 FFM API.
 *
 * <p>SafeTensors layout:
 * <pre>
 *   [8 bytes: header_size (little-endian u64)]
 *   [header_size bytes: JSON metadata]
 *   [tensor data: contiguous float32 blobs]
 * </pre>
 *
 * <p>Cross-ecosystem: weights saved here load in Python via
 * {@code safetensors.torch.load_file()}.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * StateDict.save(model.stateDict(), Path.of("model.safetensors"));
 * model.loadStateDict(StateDict.load(Path.of("model.safetensors")));
 * }</pre>
 */
public final class StateDict {

    private StateDict() {}

    // ── Save ─────────────────────────────────────────────────────────────

    public static void save(Map<String, GradTensor> state, Path path) throws IOException {
        StringBuilder json = new StringBuilder("{");
        long dataOffset = 0;
        for (var entry : state.entrySet()) {
            GradTensor t = entry.getValue();
            long byteLen = t.numel() * Float.BYTES;
            json.append('"').append(entry.getKey()).append("\":{")
                .append("\"dtype\":\"F32\",")
                .append("\"shape\":").append(shapeJson(t.shape())).append(',')
                .append("\"data_offsets\":[").append(dataOffset).append(',')
                .append(dataOffset + byteLen).append("]},");
            dataOffset += byteLen;
        }
        if (json.charAt(json.length() - 1) == ',') json.deleteCharAt(json.length() - 1);
        json.append('}');

        byte[] headerBytes = json.toString().getBytes(StandardCharsets.UTF_8);
        long headerSize = headerBytes.length;
        long totalSize  = 8 + headerSize + dataOffset;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(totalSize, 8);
            seg.set(ValueLayout.JAVA_LONG_UNALIGNED, 0, headerSize);
            MemorySegment.copy(MemorySegment.ofArray(headerBytes), 0, seg, 8, headerSize);

            long cursor = 8 + headerSize;
            for (GradTensor t : state.values()) {
                float[] data = t.data();
                MemorySegment.copy(MemorySegment.ofArray(data), 0,
                                   seg, cursor, (long) data.length * Float.BYTES);
                cursor += (long) data.length * Float.BYTES;
            }
            Files.write(path, seg.toArray(ValueLayout.JAVA_BYTE));
        }
    }

    // ── Load ─────────────────────────────────────────────────────────────

    public static Map<String, GradTensor> load(Path path) throws IOException {
        byte[] raw = Files.readAllBytes(path);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(raw.length, 8);
            MemorySegment.copy(MemorySegment.ofArray(raw), 0, seg, 0, raw.length);

            long headerSize = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, 0);
            byte[] hb = new byte[(int) headerSize];
            MemorySegment.copy(seg, 8, MemorySegment.ofArray(hb), 0, headerSize);

            return parseHeader(new String(hb, StandardCharsets.UTF_8), seg, 8 + headerSize);
        }
    }

    // ── Minimal JSON parser ───────────────────────────────────────────────

    private static Map<String, GradTensor> parseHeader(String json,
                                                        MemorySegment seg,
                                                        long dataBase) {
        Map<String, GradTensor> result = new LinkedHashMap<>();
        json = json.trim().replaceAll("^\\{|\\}$", "");
        int i = 0;
        while (i < json.length()) {
            int ks = json.indexOf('"', i) + 1; if (ks == 0) break;
            int ke = json.indexOf('"', ks);
            String key = json.substring(ks, ke);
            if (key.equals("__metadata__")) { i = json.indexOf('}', ke) + 1; continue; }

            int shapeStart = json.indexOf('[', ke) + 1;
            int shapeEnd   = json.indexOf(']', shapeStart);
            String[] sp = json.substring(shapeStart, shapeEnd).split(",");
            long[] shape = new long[sp[0].isBlank() ? 0 : sp.length];
            for (int s = 0; s < shape.length; s++) shape[s] = Long.parseLong(sp[s].trim());

            int offStart = json.indexOf('[', shapeEnd) + 1;
            int offEnd   = json.indexOf(']', offStart);
            String[] op = json.substring(offStart, offEnd).split(",");
            long start = Long.parseLong(op[0].trim()), end = Long.parseLong(op[1].trim());

            float[] data = new float[(int) ((end - start) / Float.BYTES)];
            MemorySegment.copy(seg, dataBase + start,
                               MemorySegment.ofArray(data), 0, end - start);
            result.put(key, GradTensor.of(data, shape.length == 0 ? new long[]{data.length} : shape));
            i = json.indexOf('}', offEnd) + 1;
        }
        return result;
    }

    private static String shapeJson(long[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) { if (i > 0) sb.append(','); sb.append(shape[i]); }
        return sb.append(']').toString();
    }
}
