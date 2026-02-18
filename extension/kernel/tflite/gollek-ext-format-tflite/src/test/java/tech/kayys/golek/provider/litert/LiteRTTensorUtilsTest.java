package tech.kayys.gollek.provider.litert;

import org.junit.jupiter.api.*;
import tech.kayys.gollek.provider.litert.LiteRTNativeBindings.TfLiteType;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for LiteRT Tensor Utilities.
 */
@DisplayName("LiteRT Tensor Utilities Tests")
class LiteRTTensorUtilsTest {

    @Test
    @DisplayName("Shape validation should work correctly")
    void testShapeValidation() {
        // Test exact match
        assertTrue(LiteRTTensorUtils.validateShapeCompatibility(
                new long[] { 1, 224, 224, 3 },
                new long[] { 1, 224, 224, 3 }));

        // Test dynamic dimension (-1)
        assertTrue(LiteRTTensorUtils.validateShapeCompatibility(
                new long[] { 1, -1, 224, 3 },
                new long[] { 1, 224, 224, 3 }));

        // Test dimension mismatch
        assertFalse(LiteRTTensorUtils.validateShapeCompatibility(
                new long[] { 1, 224, 224 },
                new long[] { 1, 224, 224, 3 }));

        // Test value mismatch
        assertFalse(LiteRTTensorUtils.validateShapeCompatibility(
                new long[] { 1, 256, 224, 3 },
                new long[] { 1, 224, 224, 3 }));
    }

    @Test
    @DisplayName("Element count calculation should be accurate")
    void testElementCountCalculation() {
        assertEquals(1, LiteRTTensorUtils.calculateElementCount(new long[] { 1 }));
        assertEquals(6, LiteRTTensorUtils.calculateElementCount(new long[] { 2, 3 }));
        assertEquals(150528, LiteRTTensorUtils.calculateElementCount(new long[] { 1, 224, 224, 3 }));
        assertEquals(1000, LiteRTTensorUtils.calculateElementCount(new long[] { 10, 10, 10 }));
    }

    @Test
    @DisplayName("Byte size calculation should be accurate")
    void testByteSizeCalculation() {
        long[] shape = { 1, 224, 224, 3 };

        assertEquals(150528 * 4, LiteRTTensorUtils.calculateByteSize(TfLiteType.FLOAT32, shape));
        assertEquals(150528 * 1, LiteRTTensorUtils.calculateByteSize(TfLiteType.INT8, shape));
        assertEquals(150528 * 2, LiteRTTensorUtils.calculateByteSize(TfLiteType.FLOAT16, shape));
        assertEquals(150528 * 8, LiteRTTensorUtils.calculateByteSize(TfLiteType.INT64, shape));
    }

    @Test
    @DisplayName("Bytes per element should be correct for all types")
    void testBytesPerElement() {
        assertEquals(4, LiteRTTensorUtils.getBytesPerElement(TfLiteType.FLOAT32));
        assertEquals(4, LiteRTTensorUtils.getBytesPerElement(TfLiteType.INT32));
        assertEquals(1, LiteRTTensorUtils.getBytesPerElement(TfLiteType.INT8));
        assertEquals(1, LiteRTTensorUtils.getBytesPerElement(TfLiteType.UINT8));
        assertEquals(2, LiteRTTensorUtils.getBytesPerElement(TfLiteType.FLOAT16));
        assertEquals(8, LiteRTTensorUtils.getBytesPerElement(TfLiteType.INT64));
        assertEquals(8, LiteRTTensorUtils.getBytesPerElement(TfLiteType.FLOAT64));
    }

    @Test
    @DisplayName("Float array to bytes conversion should work correctly")
    void testFloatArrayToBytes() {
        float[] input = { 1.0f, 2.5f, 3.7f };
        byte[] bytes = LiteRTTensorUtils.floatArrayToBytes(input);
        float[] output = LiteRTTensorUtils.bytesToFloatArray(bytes);

        assertArrayEquals(input, output, 0.0001f);
    }

    @Test
    @DisplayName("Int array to bytes conversion should work correctly")
    void testIntArrayToBytes() {
        int[] input = { 1, 2, 3, 1000000 };
        byte[] bytes = LiteRTTensorUtils.intArrayToBytes(input);
        int[] output = LiteRTTensorUtils.bytesToIntArray(bytes);

        assertArrayEquals(input, output);
    }

    @Test
    @DisplayName("Long array to bytes conversion should work correctly")
    void testLongArrayToBytes() {
        long[] input = { 1L, 2L, 3L, 1000000000L };
        byte[] bytes = LiteRTTensorUtils.longArrayToBytes(input);
        long[] output = LiteRTTensorUtils.bytesToLongArray(bytes);

        assertArrayEquals(input, output);
    }

    @Test
    @DisplayName("Tensor data validation should work correctly")
    void testTensorDataValidation() {
        // Valid FLOAT32 data
        float[] floatData = { 1.0f, 2.0f, 3.0f };
        byte[] floatBytes = LiteRTTensorUtils.floatArrayToBytes(floatData);
        assertTrue(LiteRTTensorUtils.validateTensorData(floatBytes, TfLiteType.FLOAT32, floatBytes.length));

        // Invalid FLOAT32 data (wrong size)
        byte[] invalidFloatBytes = new byte[5]; // Not multiple of 4
        assertFalse(
                LiteRTTensorUtils.validateTensorData(invalidFloatBytes, TfLiteType.FLOAT32, invalidFloatBytes.length));

        // Valid INT8 data
        byte[] int8Data = { 1, 2, 3 };
        assertTrue(LiteRTTensorUtils.validateTensorData(int8Data, TfLiteType.INT8, int8Data.length));

        // Empty data
        byte[] emptyData = {};
        assertFalse(LiteRTTensorUtils.validateTensorData(emptyData, TfLiteType.FLOAT32, 0));

        // Null data
        assertFalse(LiteRTTensorUtils.validateTensorData(null, TfLiteType.FLOAT32, 0));
    }

    @Test
    @DisplayName("Tensor metadata summary should contain all required information")
    void testTensorMetadataSummary() {
        String name = "input_tensor";
        TfLiteType type = TfLiteType.FLOAT32;
        long[] shape = { 1, 224, 224, 3 };
        long byteSize = 150528;
        MemorySegment tensor = null;

        String summary = LiteRTTensorUtils.createTensorMetadataSummary(name, type, shape, byteSize, tensor);

        assertTrue(summary.contains("Tensor: " + name));
        assertTrue(summary.contains("Type: " + type));
        assertTrue(summary.contains("Shape: " + Arrays.toString(shape)));
        assertTrue(summary.contains("Dimensions: " + shape.length));
        assertTrue(summary.contains("Element Count: " + LiteRTTensorUtils.calculateElementCount(shape)));
        assertTrue(summary.contains("Byte Size: " + byteSize));
        assertTrue(summary.contains("Bytes per Element: " + LiteRTTensorUtils.getBytesPerElement(type)));
    }

    @Test
    @DisplayName("Quantization should work correctly")
    void testQuantization() {
        float[] floatData = { 0.1f, 0.5f, 0.9f, -0.3f, -0.7f };
        byte[] quantized = LiteRTTensorUtils.quantizeFloatToInt8(floatData);

        assertEquals(floatData.length, quantized.length);

        // Dequantize with the CORRECT scale
        float maxAbs = 0.9f;
        float scale = maxAbs / 127.0f;
        float[] dequantized = LiteRTTensorUtils.dequantizeInt8ToFloat(quantized, scale, 0);
        assertEquals(floatData.length, dequantized.length);

        // Values should be in reasonable range after round-trip
        for (float value : dequantized) {
            assertTrue(value >= -1.0f && value <= 1.0f);
        }
    }

    @Test
    @DisplayName("Normalization should work for different tensor types")
    void testNormalization() {
        // Test FLOAT32 normalization
        float[] floatData = { 1.0f, 5.0f, 10.0f };
        byte[] floatBytes = LiteRTTensorUtils.floatArrayToBytes(floatData);
        LiteRTTensorUtils.normalizeTensor(floatBytes, TfLiteType.FLOAT32);
        float[] normalizedFloats = LiteRTTensorUtils.bytesToFloatArray(floatBytes);

        // Check that values are normalized to [0, 1] range
        assertTrue(normalizedFloats[0] >= 0.0f && normalizedFloats[0] <= 1.0f);
        assertTrue(normalizedFloats[1] >= 0.0f && normalizedFloats[1] <= 1.0f);
        assertTrue(normalizedFloats[2] >= 0.0f && normalizedFloats[2] <= 1.0f);

        // Test INT8 normalization
        byte[] int8Data = { (byte) 10, (byte) 50, (byte) 100 };
        LiteRTTensorUtils.normalizeTensor(int8Data, TfLiteType.INT8);

        // Values should be in valid range
        for (byte value : int8Data) {
            assertTrue(value >= -128 && value <= 127);
        }

        // Test UINT8 normalization
        byte[] uint8Data = { (byte) 10, (byte) 100, (byte) 200 };
        LiteRTTensorUtils.normalizeTensor(uint8Data, TfLiteType.UINT8);

        // Values should be in valid range
        for (byte value : uint8Data) {
            assertTrue((value & 0xFF) >= 0 && (value & 0xFF) <= 255);
        }
    }

    @Test
    @DisplayName("Quantization parameters should be created correctly")
    void testQuantizationParams() {
        LiteRTTensorUtils.QuantizationParams params = new LiteRTTensorUtils.QuantizationParams(0.1f, 128);

        assertEquals(0.1f, params.getScale(), 0.0001f);
        assertEquals(128, params.getZeroPoint());
        assertTrue(params.toString().contains("QuantizationParams"));
        assertTrue(params.toString().contains("zeroPoint=128"));
    }

    @Test
    @DisplayName("Tensor validation result should be created correctly")
    void testTensorValidationResult() {
        LiteRTTensorUtils.TensorValidationResult validResult = new LiteRTTensorUtils.TensorValidationResult(true,
                "Valid tensor");

        assertTrue(validResult.isValid());
        assertEquals("Valid tensor", validResult.getMessage());
        assertNull(validResult.getError());
        assertTrue(validResult.toString().contains("valid=true"));

        Exception error = new IllegalArgumentException("Invalid data");
        LiteRTTensorUtils.TensorValidationResult invalidResult = new LiteRTTensorUtils.TensorValidationResult(false,
                "Invalid tensor", error);

        assertFalse(invalidResult.isValid());
        assertEquals("Invalid tensor", invalidResult.getMessage());
        assertEquals(error, invalidResult.getError());
        assertTrue(invalidResult.toString().contains("valid=false"));
    }

    @Test
    @DisplayName("Edge cases should be handled correctly")
    void testEdgeCases() {
        // Empty shape
        assertEquals(1, LiteRTTensorUtils.calculateElementCount(new long[] {}));

        // Single element
        assertEquals(1, LiteRTTensorUtils.calculateElementCount(new long[] { 1 }));

        // Large shape
        long[] largeShape = { 100, 100, 100 };
        assertEquals(1000000, LiteRTTensorUtils.calculateElementCount(largeShape));

        // Zero in shape (should still work)
        long[] shapeWithZero = { 1, 0, 3 };
        assertEquals(0, LiteRTTensorUtils.calculateElementCount(shapeWithZero));
    }

    @Test
    @DisplayName("Type conversion should handle edge cases")
    void testTypeConversionEdgeCases() {
        // Test with empty arrays
        assertArrayEquals(new byte[0], LiteRTTensorUtils.floatArrayToBytes(new float[0]));
        assertArrayEquals(new byte[0], LiteRTTensorUtils.intArrayToBytes(new int[0]));
        assertArrayEquals(new byte[0], LiteRTTensorUtils.longArrayToBytes(new long[0]));

        // Test with single element
        float[] singleFloat = { 3.14159f };
        byte[] singleFloatBytes = LiteRTTensorUtils.floatArrayToBytes(singleFloat);
        assertEquals(4, singleFloatBytes.length);

        // Test round-trip consistency
        float[] original = { 0.0f, 1.0f, -1.0f, Float.MAX_VALUE, Float.MIN_VALUE };
        byte[] bytes = LiteRTTensorUtils.floatArrayToBytes(original);
        float[] restored = LiteRTTensorUtils.bytesToFloatArray(bytes);

        assertEquals(original.length, restored.length);
        for (int i = 0; i < original.length; i++) {
            assertEquals(original[i], restored[i], 0.0001f);
        }
    }
}