package tech.kayys.gollek.ml.nn.util.gguf;

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
 * SDK-friendly GGUF Writer.
 * Supports exporting models to GGUF (F32/F16).
 */
public final class GgufWriter {

    private static final int MAGIC = 0x46554747; // "GGUF" in LE

    public static void save(Path path, Map<String, GradTensor> tensors, Map<String, GgufMetaValue> metadata) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
             FileChannel channel = fos.getChannel()) {

            // 1. Header
            ByteBuffer headerBuf = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
            headerBuf.putInt(MAGIC);
            headerBuf.putInt(3); // Version 3
            headerBuf.putLong(tensors.size());
            headerBuf.putLong(metadata.size());
            headerBuf.flip();
            channel.write(headerBuf);

            // 2. KVs
            for (Map.Entry<String, GgufMetaValue> entry : metadata.entrySet()) {
                writeString(channel, entry.getKey());
                writeValue(channel, entry.getValue());
            }

            // 3. Tensor infos
            long currentOffset = 0;
            for (Map.Entry<String, GradTensor> entry : tensors.entrySet()) {
                String name = entry.getKey();
                GradTensor tensor = entry.getValue();
                
                writeString(channel, name);
                
                long[] shape = tensor.shape();
                writeInt(channel, shape.length);
                for (long d : shape) writeLong(channel, d);
                
                GgmlType type = GgmlType.F32; // Default for SDK export
                writeInt(channel, type.id);
                writeLong(channel, currentOffset);
                
                currentOffset += type.bytesFor(tensor.numel());
                // Align offset
                currentOffset = (currentOffset + 31) & ~31;
            }

            // 4. Alignment padding
            long pos = channel.position();
            int alignment = 32;
            long dataStart = (pos + alignment - 1) & ~(alignment - 1);
            if (dataStart > pos) {
                channel.write(ByteBuffer.allocate((int) (dataStart - pos)));
            }

            // 5. Data
            for (GradTensor tensor : tensors.values()) {
                float[] data = tensor.data();
                ByteBuffer dataBuf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                for (float f : data) dataBuf.putFloat(f);
                dataBuf.flip();
                channel.write(dataBuf);
                
                // Align data segment
                long endPos = channel.position();
                long nextAligned = (endPos + alignment - 1) & ~(alignment - 1);
                if (nextAligned > endPos) {
                    channel.write(ByteBuffer.allocate((int) (nextAligned - endPos)));
                }
            }
        }
    }

    private static void writeString(FileChannel channel, String s) throws IOException {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        writeLong(channel, bytes.length);
        channel.write(ByteBuffer.wrap(bytes));
    }

    private static void writeInt(FileChannel channel, int v) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(v);
        buf.flip();
        channel.write(buf);
    }

    private static void writeLong(FileChannel channel, long v) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(v);
        buf.flip();
        channel.write(buf);
    }

    private static void writeValue(FileChannel channel, GgufMetaValue val) throws IOException {
        if (val instanceof GgufMetaValue.StringVal s) {
            writeInt(channel, GgufMetaType.STRING.id);
            writeString(channel, s.value());
        } else if (val instanceof GgufMetaValue.Uint32Val u) {
            writeInt(channel, GgufMetaType.UINT32.id);
            writeInt(channel, (int) u.value());
        }
        // ... extend as needed
    }
}
