package tech.kayys.gollek.inference.libtorch.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SafetensorsLoaderTest {

    @TempDir
    Path tempDir;

    @Test
    void inspectParsesSafetensorsMetadata() throws Exception {
        Path file = tempDir.resolve("mini.safetensors");
        writeMiniSafetensors(file);

        SafetensorsLoader loader = new SafetensorsLoader();
        setField(loader, "headerParser", new SafetensorsHeaderParser());

        Map<String, SafetensorsHeaderParser.TensorMetadata> metadata = loader.inspect(file);
        assertEquals(1, metadata.size());
        assertTrue(metadata.containsKey("linear.weight"));

        SafetensorsHeaderParser.TensorMetadata tensor = metadata.get("linear.weight");
        assertEquals("F32", tensor.dtype());
        assertArrayEquals(new long[] { 2 }, tensor.shape());
        assertEquals(8L, tensor.length()); // 2 x f32
    }

    @Test
    void headerParserCalculatesAbsoluteOffsets() throws Exception {
        Path file = tempDir.resolve("offsets.safetensors");
        long[] expected = writeMiniSafetensors(file);

        SafetensorsHeaderParser parser = new SafetensorsHeaderParser();
        Map<String, SafetensorsHeaderParser.TensorMetadata> metadata = parser.parse(file);
        SafetensorsHeaderParser.TensorMetadata tensor = metadata.get("linear.weight");

        assertEquals(expected[0], tensor.absoluteStart());
        assertEquals(expected[1], tensor.length());
    }

    private static void setField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    /**
     * Writes a minimal valid .safetensors file with one tensor:
     * linear.weight: dtype F32, shape [2], values [1.5, -2.0]
     *
     * @return [absoluteStart, length]
     */
    private static long[] writeMiniSafetensors(Path path) throws IOException {
        String headerJson = "{\"linear.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        byte[] headerBytes = headerJson.getBytes(StandardCharsets.UTF_8);

        ByteBuffer prefix = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        prefix.putLong(headerBytes.length);

        ByteBuffer payload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        payload.putFloat(1.5f);
        payload.putFloat(-2.0f);

        byte[] fileBytes = new byte[8 + headerBytes.length + 8];
        System.arraycopy(prefix.array(), 0, fileBytes, 0, 8);
        System.arraycopy(headerBytes, 0, fileBytes, 8, headerBytes.length);
        System.arraycopy(payload.array(), 0, fileBytes, 8 + headerBytes.length, 8);

        Files.write(path, fileBytes);
        return new long[] { 8L + headerBytes.length, 8L };
    }
}
