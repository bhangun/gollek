package tech.kayys.golek.provider.litert;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import tech.kayys.golek.provider.core.spi.ErrorCode;

/**
 * Utility for converting between platform tensors and native tensor formats.
 * 
 * @author bhangun
 * @since 1.0.0
 */
public class TensorConverter {

    /**
     * Convert TensorData to byte array for native format.
     */
    public static byte[] toNativeBytes(TensorData tensor) {
        if (tensor.getData() != null) {
            return tensor.getData();
        }

        // Convert from typed data to bytes
        return switch (tensor.getDtype()) {
            case FLOAT32 -> float32ToBytes(tensor.getFloatData());
            case FLOAT16 -> float16ToBytes(tensor.getFloatData());
            case INT8 -> int8ToBytes(tensor.getIntData());
            case UINT8 -> uint8ToBytes(tensor.getIntData());
            case INT16 -> int16ToBytes(tensor.getIntData());
            case INT32 -> int32ToBytes(tensor.getIntData());
            case INT64 -> int64ToBytes(tensor.getLongData());
            case BOOL -> boolToBytes(tensor.getBoolData());
            default -> throw new TensorException(
                    ErrorCode.TENSOR_INVALID_DATA,
                    "Unsupported tensor type: " + tensor.getDtype(),
                    tensor.getName());
        };
    }

    /**
     * Convert native bytes to TensorData.
     */
    public static TensorData fromNativeBytes(
            byte[] data,
            TensorDataType dtype,
            long[] shape,
            String name) {

        return TensorData.builder()
                .data(data)
                .dtype(dtype)
                .shape(shape)
                .name(name)
                .build();
    }

    // ===== Type Conversion Methods =====

    private static byte[] float32ToBytes(float[] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 4);
        buffer.order(ByteOrder.nativeOrder());
        for (float f : data) {
            buffer.putFloat(f);
        }
        return buffer.array();
    }

    private static byte[] float16ToBytes(float[] data) {
        // FP16 conversion (simplified - use proper library in production)
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 2);
        buffer.order(ByteOrder.nativeOrder());
        for (float f : data) {
            buffer.putShort((short) Float.floatToFloat16(f));
        }
        return buffer.array();
    }

    private static byte[] int8ToBytes(int[] data) {
        byte[] bytes = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            bytes[i] = (byte) data[i];
        }
        return bytes;
    }

    private static byte[] uint8ToBytes(int[] data) {
        byte[] bytes = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            bytes[i] = (byte) (data[i] & 0xFF);
        }
        return bytes;
    }

    private static byte[] int16ToBytes(int[] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 2);
        buffer.order(ByteOrder.nativeOrder());
        for (int i : data) {
            buffer.putShort((short) i);
        }
        return buffer.array();
    }

    private static byte[] int32ToBytes(int[] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 4);
        buffer.order(ByteOrder.nativeOrder());
        for (int i : data) {
            buffer.putInt(i);
        }
        return buffer.array();
    }

    private static byte[] int64ToBytes(long[] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 8);
        buffer.order(ByteOrder.nativeOrder());
        for (long l : data) {
            buffer.putLong(l);
        }
        return buffer.array();
    }

    private static byte[] boolToBytes(boolean[] data) {
        byte[] bytes = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            bytes[i] = data[i] ? (byte) 1 : (byte) 0;
        }
        return bytes;
    }

    /**
     * Calculate total number of elements in a tensor.
     */
    public static long calculateElements(long[] shape) {
        long total = 1;
        for (long dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * Calculate byte size for a tensor.
     */
    public static long calculateByteSize(long[] shape, TensorDataType dtype) {
        long elements = calculateElements(shape);
        int bytesPerElement = switch (dtype) {
            case FLOAT32, INT32, UINT32 -> 4;
            case FLOAT64, INT64, UINT64 -> 8;
            case FLOAT16, INT16, UINT16 -> 2;
            case INT8, UINT8, BOOL -> 1;
            default -> 4;
        };
        return elements * bytesPerElement;
    }
}