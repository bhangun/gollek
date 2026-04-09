package tech.kayys.gollek.ml.nn.util;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Map;

/**
 * Utility for writing tensors in the Safetensor format.
 * <p>
 * Safetensors format:
 * <ul>
 *   <li><b>Header Length:</b> 8-byte Little-Endian unsigned 64-bit integer N</li>
 *   <li><b>Header:</b> N bytes of UTF-8 encoded JSON string</li>
 *   <li><b>Data:</b> Raw binary data stored at offsets specified in the header</li>
 * </ul>
 */
public class SafetensorWriter {

    private static final Gson GSON = new GsonBuilder().create();

    /**
     * Save a map of tensors to a Safetensor file.
     *
     * @param path    the destination path
     * @param tensors named tensors to save
     * @throws IOException if writing fails
     */
    public static void save(Path path, Map<String, GradTensor> tensors) throws IOException {
        JsonObject header = new JsonObject();
        long currentOffset = 0;

        // 1. Build JSON metadata for each tensor
        for (Map.Entry<String, GradTensor> entry : tensors.entrySet()) {
            String name = entry.getKey();
            GradTensor tensor = entry.getValue();

            JsonObject tensorMeta = new JsonObject();
            
            // Map DType to Safetensor string
            String dtypeStr = switch (tensor.dtype()) {
                case FLOAT32 -> "F32";
                case FLOAT16 -> "F16";
                case BFLOAT16 -> "BF16";
                case INT32 -> "I32";
                case INT64 -> "I64";
                case INT8 -> "I8";
                default -> "F32"; // Fallback to F32 for GradTensor current storage
            };
            tensorMeta.addProperty("dtype", dtypeStr);
            
            com.google.gson.JsonArray shapeArr = new com.google.gson.JsonArray();
            for (long d : tensor.shape()) {
                shapeArr.add(d);
            }
            tensorMeta.add("shape", shapeArr);

            long numel = tensor.numel();
            long byteSize = numel * 4; // Each F32 element is 4 bytes
            
            com.google.gson.JsonArray offsets = new com.google.gson.JsonArray();
            offsets.add(currentOffset);
            offsets.add(currentOffset + byteSize);
            tensorMeta.add("data_offsets", offsets);

            header.add(name, tensorMeta);
            currentOffset += byteSize;
        }

        // 2. Generate JSON string and handle alignment
        String jsonString = GSON.toJson(header);
        byte[] headerBytes = jsonString.getBytes(StandardCharsets.UTF_8);
        int headerLen = headerBytes.length;
        
        // Standard Safetensors recommendation: data segment should start at 8-byte aligned offset.
        // data_start = 8 (for header link) + headerLen
        int alignmentPadding = (int) ((8 - (8 + headerLen) % 8) % 8);
        if (alignmentPadding > 0) {
            // Pad the JSON with space characters to reach alignment
            StringBuilder sb = new StringBuilder(jsonString);
            for (int i = 0; i < alignmentPadding; i++) {
                sb.append(" ");
            }
            headerBytes = sb.toString().getBytes(StandardCharsets.UTF_8);
            headerLen = headerBytes.length;
        }

        // 3. Write to file
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
             FileChannel channel = fos.getChannel()) {
            
            // Write 8-byte LE header length
            ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            lenBuf.putLong(headerLen);
            lenBuf.flip();
            while (lenBuf.hasRemaining()) {
                channel.write(lenBuf);
            }

            // Write JSON header
            channel.write(ByteBuffer.wrap(headerBytes));

            // Write tensor data segments in order
            // We use a buffer to convert float array to Little-Endian bytes efficiently
            ByteBuffer dataBuf = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN);
            for (GradTensor tensor : tensors.values()) {
                float[] data = tensor.data();
                for (float f : data) {
                    if (!dataBuf.hasRemaining()) {
                        dataBuf.flip();
                        while (dataBuf.hasRemaining()) {
                            channel.write(dataBuf);
                        }
                        dataBuf.clear();
                    }
                    dataBuf.putFloat(f);
                }
            }
            
            // Final flush of data buffer
            dataBuf.flip();
            while (dataBuf.hasRemaining()) {
                channel.write(dataBuf);
            }
        }
    }
}
