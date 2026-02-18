import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.model.ModelFormat;

import org.junit.jupiter.api.DisplayName;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ModelFormat enum.
 *
 * @author Bhangun
 */
class ModelFormatTest {

    @Test
    @DisplayName("Should have correct properties for PyTorch format")
    void testPyTorchProperties() {
        ModelFormat pytorch = ModelFormat.PYTORCH;

        assertEquals("pytorch", pytorch.getId());
        assertEquals("PyTorch", pytorch.getDisplayName());
        assertTrue(pytorch.isConvertible());
        assertTrue(pytorch.requiresConversion());
        assertTrue(pytorch.getFileExtensions().contains(".bin"));
        assertTrue(pytorch.getFileExtensions().contains(".pt"));
        assertTrue(pytorch.getFileExtensions().contains(".pth"));
        assertTrue(pytorch.getMarkerFiles().contains("pytorch_model.bin"));
    }

    @Test
    @DisplayName("Should have correct properties for SafeTensors format")
    void testSafeTensorsProperties() {
        ModelFormat safetensors = ModelFormat.SAFETENSORS;

        assertEquals("safetensors", safetensors.getId());
        assertEquals("SafeTensors", safetensors.getDisplayName());
        assertTrue(safetensors.isConvertible());
        assertTrue(safetensors.requiresConversion());
        assertTrue(safetensors.getFileExtensions().contains(".safetensors"));
        assertTrue(safetensors.getMarkerFiles().contains("model.safetensors"));
    }

    @Test
    @DisplayName("Should have correct properties for GGUF format")
    void testGGUFProperties() {
        ModelFormat gguf = ModelFormat.GGUF;

        assertEquals("gguf", gguf.getId());
        assertEquals("GGUF", gguf.getDisplayName());
        assertFalse(gguf.isConvertible()); // Already in GGUF format
        assertFalse(gguf.requiresConversion());
        assertTrue(gguf.getFileExtensions().contains(".gguf"));
    }

    @Test
    @DisplayName("Should find format by ID")
    void testFromId() {
        assertEquals(ModelFormat.PYTORCH, ModelFormat.fromId("pytorch"));
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromId("safetensors"));
        assertEquals(ModelFormat.GGUF, ModelFormat.fromId("gguf"));
        assertEquals(ModelFormat.UNKNOWN, ModelFormat.fromId("invalid"));
        assertEquals(ModelFormat.UNKNOWN, ModelFormat.fromId(null));
        assertEquals(ModelFormat.UNKNOWN, ModelFormat.fromId(""));
    }

    @Test
    @DisplayName("Should find format by file extension")
    void testFromExtension() {
        assertEquals(ModelFormat.PYTORCH, ModelFormat.fromExtension(".bin"));
        assertEquals(ModelFormat.PYTORCH, ModelFormat.fromExtension("bin")); // Without dot
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromExtension(".safetensors"));
        assertEquals(ModelFormat.GGUF, ModelFormat.fromExtension(".gguf"));
        assertEquals(ModelFormat.ONNX, ModelFormat.fromExtension(".onnx"));
        assertEquals(ModelFormat.UNKNOWN, ModelFormat.fromExtension(".xyz"));
        assertEquals(ModelFormat.UNKNOWN, ModelFormat.fromExtension(null));
    }

    @Test
    @DisplayName("Should identify convertible formats")
    void testConvertibleFormats() {
        Set<ModelFormat> convertible = ModelFormat.getConvertibleFormats();

        assertTrue(convertible.contains(ModelFormat.PYTORCH));
        assertTrue(convertible.contains(ModelFormat.SAFETENSORS));
        assertTrue(convertible.contains(ModelFormat.TENSORFLOW));
        assertTrue(convertible.contains(ModelFormat.FLAX));
        assertFalse(convertible.contains(ModelFormat.GGUF)); // Already in GGUF
        assertFalse(convertible.contains(ModelFormat.UNKNOWN));
    }

    @Test
    @DisplayName("Should handle case-insensitive IDs")
    void testCaseInsensitiveId() {
        assertEquals(ModelFormat.PYTORCH, ModelFormat.fromId("PYTORCH"));
        assertEquals(ModelFormat.PYTORCH, ModelFormat.fromId("PyTorch"));
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromId("SAFETENSORS"));
    }

    @Test
    @DisplayName("Should handle case-insensitive extensions")
    void testCaseInsensitiveExtension() {
        assertEquals(ModelFormat.PYTORCH, ModelFormat.fromExtension(".BIN"));
        assertEquals(ModelFormat.PYTORCH, ModelFormat.fromExtension("BIN"));
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromExtension(".SAFETENSORS"));
    }

    @Test
    @DisplayName("Should identify convertible status correctly")
    void testConvertibleStatus() {
        assertTrue(ModelFormat.PYTORCH.isConvertible());
        assertTrue(ModelFormat.SAFETENSORS.isConvertible());
        assertTrue(ModelFormat.TENSORFLOW.isConvertible());
        assertTrue(ModelFormat.FLAX.isConvertible());
        assertFalse(ModelFormat.GGUF.isConvertible());
        assertFalse(ModelFormat.UNKNOWN.isConvertible());
    }

    @Test
    @DisplayName("Should identify requires conversion status correctly")
    void testRequiresConversionStatus() {
        assertTrue(ModelFormat.PYTORCH.requiresConversion());
        assertTrue(ModelFormat.SAFETENSORS.requiresConversion());
        assertTrue(ModelFormat.TENSORFLOW.requiresConversion());
        assertTrue(ModelFormat.FLAX.requiresConversion());
        assertFalse(ModelFormat.GGUF.requiresConversion());
        assertFalse(ModelFormat.UNKNOWN.requiresConversion());
    }

    @Test
    @DisplayName("Should handle all format constants")
    void testAllFormatConstants() {
        ModelFormat[] allFormats = ModelFormat.values();

        assertEquals(6, allFormats.length); // PYTORCH, SAFETENSORS, TENSORFLOW, FLAX, ONNX, GGUF, UNKNOWN

        // Verify each format has unique ID
        for (int i = 0; i < allFormats.length; i++) {
            for (int j = i + 1; j < allFormats.length; j++) {
                assertNotEquals(allFormats[i].getId(), allFormats[j].getId(),
                        "Format IDs should be unique");
            }
        }
    }
}