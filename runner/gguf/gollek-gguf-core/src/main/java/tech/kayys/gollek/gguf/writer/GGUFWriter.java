package tech.kayys.gollek.gguf.writer;

import tech.kayys.gollek.gguf.core.GgmlType;
import tech.kayys.gollek.gguf.core.GgufMetaType;
import tech.kayys.gollek.gguf.core.GgufMetaValue;
import tech.kayys.gollek.gguf.core.GgufModel;
import tech.kayys.alkhawarizm.core.tensor.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * GGUF v3 writer for Tensor-backed training models.
 */
public final class GGUFWriter {

    public enum TensorEncoding {
        F32,
        F16,
        Q8_0,
        Q4_0
    }

    private record EncodedTensor(
            String name,
            long[] shape,
            GgmlType type,
            long offset,
            byte[] data) {
    }

    private GGUFWriter() {
    }

    public static void save(Path path, Map<String, Tensor> tensors, Map<String, GgufMetaValue> metadata)
            throws IOException {
        save(path, tensors, metadata, TensorEncoding.F32);
    }

    public static void save(
            Path path,
            Map<String, Tensor> tensors,
            Map<String, GgufMetaValue> metadata,
            TensorEncoding encoding) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(tensors, "tensors");
        Objects.requireNonNull(metadata, "metadata");
        Objects.requireNonNull(encoding, "encoding");

        Map<String, GgufMetaValue> orderedMetadata = new LinkedHashMap<>(metadata);
        int alignment = alignment(orderedMetadata);
        List<EncodedTensor> encodedTensors = encodeTensors(tensors, encoding, alignment);

        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        try (FileChannel channel = FileChannel.open(
                path,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE)) {
            writeHeader(channel, encodedTensors.size(), orderedMetadata.size());
            writeMetadata(channel, orderedMetadata);
            writeTensorInfos(channel, encodedTensors);
            writePadding(channel, alignment);
            writeTensorData(channel, encodedTensors, alignment);
        }
    }

    private static List<EncodedTensor> encodeTensors(
            Map<String, Tensor> tensors,
            TensorEncoding encoding,
            int alignment) throws IOException {
        List<EncodedTensor> encoded = new ArrayList<>(tensors.size());
        long offset = 0;
        for (Map.Entry<String, Tensor> entry : tensors.entrySet()) {
            String name = requireTensorName(entry.getKey());
            Tensor tensor = Objects.requireNonNull(entry.getValue(), "tensor " + name);
            long[] shape = ggufShape(tensor.shape().dims());
            GgmlType type = tensorType(tensor, encoding);
            byte[] data = encodeTensor(tensor, type);
            encoded.add(new EncodedTensor(name, shape, type, offset, data));
            offset = alignUp(offset + data.length, alignment);
        }
        return encoded;
    }

    private static String requireTensorName(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("GGUF tensor names must not be blank");
        }
        return name;
    }

    private static long[] ggufShape(long[] shape) {
        if (shape.length == 0) {
            return new long[] { 1 };
        }
        if (shape.length > 4) {
            throw new IllegalArgumentException("GGUF supports tensors with at most 4 dimensions");
        }
        long[] copy = shape.clone();
        for (long dim : copy) {
            if (dim <= 0) {
                throw new IllegalArgumentException("GGUF tensor dimensions must be positive");
            }
        }
        return copy;
    }

    private static GgmlType tensorType(Tensor tensor, TensorEncoding encoding) {
        long n = tensor.numel();
        return switch (encoding) {
            case F32 -> GgmlType.F32;
            case F16 -> GgmlType.F16;
            case Q8_0 -> n % GgmlType.Q8_0.blockSize == 0 ? GgmlType.Q8_0 : GgmlType.F32;
            case Q4_0 -> n % GgmlType.Q4_0.blockSize == 0 ? GgmlType.Q4_0 : GgmlType.F32;
        };
    }

    private static byte[] encodeTensor(Tensor tensor, GgmlType type) throws IOException {
        return switch (type) {
            case F32 -> encodeF32(tensor.toFloatArray());
            case F16 -> encodeF16(tensor.toFloatArray());
            case Q8_0 -> encodeQ8_0(tensor.toFloatArray());
            case Q4_0 -> encodeQ4_0(tensor.toFloatArray());
            default -> throw new IllegalArgumentException("Unsupported GGUF export tensor type: " + type);
        };
    }

    private static byte[] encodeF32(float[] values) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(Math.multiplyExact(values.length, Float.BYTES));
        for (float value : values) {
            writeFloatLE(out, value);
        }
        return out.toByteArray();
    }

    private static byte[] encodeF16(float[] values) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(Math.multiplyExact(values.length, Short.BYTES));
        for (float value : values) {
            writeShortLE(out, floatToHalfBits(value));
        }
        return out.toByteArray();
    }

    private static byte[] encodeQ8_0(float[] values) throws IOException {
        if (values.length % GgmlType.Q8_0.blockSize != 0) {
            throw new IllegalArgumentException("Q8_0 tensors must be divisible by 32 elements");
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream((values.length / 32) * GgmlType.Q8_0.typeSize);
        for (int block = 0; block < values.length; block += 32) {
            float scale = maxAbs(values, block, 32) / 127.0f;
            writeShortLE(out, floatToHalfBits(scale));
            for (int i = 0; i < 32; i++) {
                int q = scale == 0.0f ? 0 : Math.round(values[block + i] / scale);
                out.write(clamp(q, -127, 127) & 0xFF);
            }
        }
        return out.toByteArray();
    }

    private static byte[] encodeQ4_0(float[] values) throws IOException {
        if (values.length % GgmlType.Q4_0.blockSize != 0) {
            throw new IllegalArgumentException("Q4_0 tensors must be divisible by 32 elements");
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream((values.length / 32) * GgmlType.Q4_0.typeSize);
        for (int block = 0; block < values.length; block += 32) {
            float scale = maxAbs(values, block, 32) / 7.0f;
            writeShortLE(out, floatToHalfBits(scale));
            for (int i = 0; i < 16; i++) {
                int lo = quantize4(values[block + i], scale) + 8;
                int hi = quantize4(values[block + 16 + i], scale) + 8;
                out.write((hi << 4) | lo);
            }
        }
        return out.toByteArray();
    }

    private static float maxAbs(float[] values, int start, int length) {
        float max = 0.0f;
        for (int i = 0; i < length; i++) {
            max = Math.max(max, Math.abs(values[start + i]));
        }
        return max;
    }

    private static int quantize4(float value, float scale) {
        if (scale == 0.0f) {
            return 0;
        }
        return clamp(Math.round(value / scale), -8, 7);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void writeHeader(FileChannel channel, int tensorCount, int metadataCount) throws IOException {
        channel.write(ByteBuffer.wrap(GgufModel.MAGIC_BYTES));
        writeIntLE(channel, GgufModel.VERSION);
        writeLongLE(channel, tensorCount);
        writeLongLE(channel, metadataCount);
    }

    private static void writeMetadata(FileChannel channel, Map<String, GgufMetaValue> metadata) throws IOException {
        for (Map.Entry<String, GgufMetaValue> entry : metadata.entrySet()) {
            writeString(channel, entry.getKey());
            writeValue(channel, entry.getValue());
        }
    }

    private static void writeTensorInfos(FileChannel channel, List<EncodedTensor> tensors) throws IOException {
        for (EncodedTensor tensor : tensors) {
            writeString(channel, tensor.name());
            writeIntLE(channel, tensor.shape().length);
            for (long dim : tensor.shape()) {
                writeLongLE(channel, dim);
            }
            writeIntLE(channel, tensor.type().id);
            writeLongLE(channel, tensor.offset());
        }
    }

    private static void writeTensorData(FileChannel channel, List<EncodedTensor> tensors, int alignment)
            throws IOException {
        for (EncodedTensor tensor : tensors) {
            channel.write(ByteBuffer.wrap(tensor.data()));
            writePadding(channel, alignment);
        }
    }

    private static void writeValue(FileChannel channel, GgufMetaValue value) throws IOException {
        writeIntLE(channel, value.type().id);
        writeValuePayload(channel, value);
    }

    private static void writeValuePayload(FileChannel channel, GgufMetaValue value) throws IOException {
        switch (value) {
            case GgufMetaValue.UInt8Val v -> writeByte(channel, v.value());
            case GgufMetaValue.Int8Val v -> writeByte(channel, v.value());
            case GgufMetaValue.UInt16Val v -> writeShortLE(channel, v.value());
            case GgufMetaValue.Int16Val v -> writeShortLE(channel, v.value());
            case GgufMetaValue.UInt32Val v -> writeIntLE(channel, (int) v.value());
            case GgufMetaValue.Int32Val v -> writeIntLE(channel, v.value());
            case GgufMetaValue.Float32Val v -> writeFloatLE(channel, v.value());
            case GgufMetaValue.BoolVal v -> writeByte(channel, v.value() ? 1 : 0);
            case GgufMetaValue.StringVal v -> writeString(channel, v.value());
            case GgufMetaValue.UInt64Val v -> writeLongLE(channel, v.value());
            case GgufMetaValue.Int64Val v -> writeLongLE(channel, v.value());
            case GgufMetaValue.Float64Val v -> writeDoubleLE(channel, v.value());
            case GgufMetaValue.ArrayVal v -> {
                writeIntLE(channel, v.elementType().id);
                writeLongLE(channel, v.elements().size());
                for (GgufMetaValue element : v.elements()) {
                    writeArrayElement(channel, v.elementType(), element);
                }
            }
        }
    }

    private static void writeArrayElement(FileChannel channel, GgufMetaType elementType, GgufMetaValue element)
            throws IOException {
        if (element.type() != elementType) {
            throw new IllegalArgumentException(
                    "GGUF array element type mismatch: expected " + elementType + ", got " + element.type());
        }
        writeValuePayload(channel, element);
    }

    private static void writeString(FileChannel channel, String value) throws IOException {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        writeLongLE(channel, bytes.length);
        channel.write(ByteBuffer.wrap(bytes));
    }

    private static int alignment(Map<String, GgufMetaValue> metadata) {
        GgufMetaValue value = metadata.getOrDefault(
                "general.alignment",
                GgufMetaValue.ofUInt32(GgufModel.DEFAULT_ALIGNMENT));
        if (value instanceof GgufMetaValue.UInt32Val uint32) {
            return Math.toIntExact(uint32.value());
        }
        if (value instanceof GgufMetaValue.Int32Val int32) {
            return int32.value();
        }
        throw new IllegalArgumentException("general.alignment must be an INT32 or UINT32 GGUF metadata value");
    }

    private static void writePadding(FileChannel channel, int alignment) throws IOException {
        long position = channel.position();
        long padding = alignUp(position, alignment) - position;
        if (padding <= 0) {
            return;
        }
        channel.write(ByteBuffer.allocate(Math.toIntExact(padding)));
    }

    private static long alignUp(long value, int alignment) {
        if (alignment <= 0) {
            throw new IllegalArgumentException("alignment must be positive");
        }
        long remainder = value % alignment;
        return remainder == 0 ? value : value + alignment - remainder;
    }

    private static void writeByte(FileChannel channel, int value) throws IOException {
        channel.write(ByteBuffer.wrap(new byte[] { (byte) value }));
    }

    private static void writeShortLE(FileChannel channel, int value) throws IOException {
        channel.write(ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN).putShort((short) value).flip());
    }

    private static void writeIntLE(FileChannel channel, int value) throws IOException {
        channel.write(ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN).putInt(value).flip());
    }

    private static void writeLongLE(FileChannel channel, long value) throws IOException {
        channel.write(ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).putLong(value).flip());
    }

    private static void writeFloatLE(FileChannel channel, float value) throws IOException {
        channel.write(ByteBuffer.allocate(Float.BYTES).order(ByteOrder.LITTLE_ENDIAN).putFloat(value).flip());
    }

    private static void writeDoubleLE(FileChannel channel, double value) throws IOException {
        channel.write(ByteBuffer.allocate(Double.BYTES).order(ByteOrder.LITTLE_ENDIAN).putDouble(value).flip());
    }

    private static void writeShortLE(OutputStream out, int value) throws IOException {
        out.write(value & 0xFF);
        out.write((value >>> 8) & 0xFF);
    }

    private static void writeFloatLE(OutputStream out, float value) throws IOException {
        int bits = Float.floatToIntBits(value);
        out.write(bits & 0xFF);
        out.write((bits >>> 8) & 0xFF);
        out.write((bits >>> 16) & 0xFF);
        out.write((bits >>> 24) & 0xFF);
    }

    private static int floatToHalfBits(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;

        if (exponent >= 31) {
            return sign | 0x7C00 | (mantissa == 0 ? 0 : 1);
        }
        if (exponent <= 0) {
            if (exponent < -10) {
                return sign;
            }
            mantissa |= 0x800000;
            int shift = 14 - exponent;
            int halfMantissa = mantissa >> shift;
            if (((mantissa >> (shift - 1)) & 1) != 0) {
                halfMantissa++;
            }
            return sign | halfMantissa;
        }

        int half = sign | (exponent << 10) | (mantissa >> 13);
        if ((mantissa & 0x1000) != 0) {
            half++;
        }
        return half;
    }
}
