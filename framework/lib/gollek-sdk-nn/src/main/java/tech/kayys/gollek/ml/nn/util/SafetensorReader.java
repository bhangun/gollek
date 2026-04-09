package tech.kayys.gollek.ml.nn.util;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Utility for reading tensors in the Safetensor format.
 */
public class SafetensorReader {

    /**
     * Read Safetensors file and return a map of tensor names to their float data.
     *
     * @param path the path to the .safetensors file
     * @return map of tensor names to float arrays
     * @throws IOException if reading fails
     */
    public static Map<String, float[]> read(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            // 1. Read header length (8 bytes LE)
            ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            if (channel.read(lenBuf) != 8) {
                throw new IOException("Failed to read Safetensor header length");
            }
            lenBuf.flip();
            long headerLen = lenBuf.getLong();

            // 2. Read JSON header
            ByteBuffer headerBuf = ByteBuffer.allocate((int) headerLen);
            if (channel.read(headerBuf) != headerLen) {
                throw new IOException("Failed to read Safetensor header content");
            }
            headerBuf.flip();
            String json = StandardCharsets.UTF_8.decode(headerBuf).toString();

            // 3. Parse JSON
            JsonObject root = JsonParser.parseString(json).getAsJsonObject();
            Map<String, float[]> tensors = new LinkedHashMap<>();

            for (Map.Entry<String, JsonElement> entry : root.entrySet()) {
                if (entry.getKey().equals("__metadata__")) continue;

                JsonObject meta = entry.getValue().getAsJsonObject();
                JsonArray offsets = meta.getAsJsonArray("data_offsets");
                long start = offsets.get(0).getAsLong();
                long end = offsets.get(1).getAsLong();
                int size = (int) (end - start);
                int numElements = size / 4;

                // 4. Read tensor data
                ByteBuffer dataBuf = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
                channel.position(8 + headerLen + start);
                if (channel.read(dataBuf) != size) {
                    throw new IOException("Failed to read data for tensor: " + entry.getKey());
                }
                dataBuf.flip();

                float[] data = new float[numElements];
                for (int i = 0; i < numElements; i++) {
                    data[i] = dataBuf.getFloat();
                }
                tensors.put(entry.getKey(), data);
            }

            return tensors;
        }
    }
}
