package tech.kayys.gollek.inference.libtorch.util;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.json.Json;
import jakarta.json.JsonObject;
import jakarta.json.JsonReader;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.io.StringReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

/**
 * Parses the header of a <code>.safetensors</code> file.
 * <p>
 * The safetensors binary format is:
 * 
 * <pre>
 *   [8 bytes LE: header_length]
 *   [header_length bytes: JSON header]
 *   [tensor data...]
 * </pre>
 * 
 * The JSON header maps tensor names to their metadata
 * (dtype, shape, data_offsets relative to the end of the header).
 */
@ApplicationScoped
public class SafetensorsHeaderParser {

    private static final Logger log = Logger.getLogger(SafetensorsHeaderParser.class);

    /**
     * Parse the safetensors header and return metadata for all tensors.
     *
     * @param path path to the .safetensors file
     * @return map of tensor name → metadata
     * @throws IOException if parsing fails
     */
    public Map<String, TensorMetadata> parse(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            // 1. Read 8-byte little-endian header size
            ByteBuffer sizeBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(sizeBuf);
            sizeBuf.flip();
            long headerLength = sizeBuf.getLong();

            if (headerLength <= 0 || headerLength > 100_000_000) {
                throw new IOException("Invalid safetensors header length: " + headerLength);
            }

            // 2. Read JSON header
            ByteBuffer headerBuf = ByteBuffer.allocate((int) headerLength);
            channel.read(headerBuf);
            headerBuf.flip();
            String json = StandardCharsets.UTF_8.decode(headerBuf).toString();

            // 3. Parse JSON and extract tensor metadata
            long dataBaseOffset = 8 + headerLength;
            return parseHeader(json, dataBaseOffset);
        }
    }

    private Map<String, TensorMetadata> parseHeader(String json, long dataBaseOffset) {
        Map<String, TensorMetadata> result = new HashMap<>();

        try (JsonReader reader = Json.createReader(new StringReader(json))) {
            JsonObject root = reader.readObject();

            for (String key : root.keySet()) {
                // Skip the __metadata__ entry
                if ("__metadata__".equals(key))
                    continue;

                JsonObject entry = root.getJsonObject(key);
                String dtype = entry.getString("dtype");
                long[] shape = entry.getJsonArray("shape").stream()
                        .mapToLong(v -> ((jakarta.json.JsonNumber) v).longValue())
                        .toArray();

                var offsets = entry.getJsonArray("data_offsets");
                long start = ((jakarta.json.JsonNumber) offsets.get(0)).longValue();
                long end = ((jakarta.json.JsonNumber) offsets.get(1)).longValue();

                result.put(key, new TensorMetadata(dtype, shape, start, end, dataBaseOffset));
            }
        }

        log.debugf("Parsed safetensors header: %d tensors (dataBaseOffset=%d)", result.size(), dataBaseOffset);
        return result;
    }

    /**
     * Metadata for a single tensor within a safetensors file.
     *
     * @param dtype      safetensors dtype string (e.g. "F32", "F16", "BF16", "I64")
     * @param shape      tensor shape
     * @param start      start offset relative to data section
     * @param end        end offset relative to data section
     * @param baseOffset absolute offset where tensor data begins in the file (8 +
     *                   headerLength)
     */
    public record TensorMetadata(String dtype, long[] shape, long start, long end, long baseOffset) {
        /** Absolute byte offset of this tensor in the file. */
        public long absoluteStart() {
            return baseOffset + start;
        }

        /** Length of the tensor data in bytes. */
        public long length() {
            return end - start;
        }
    }
}
