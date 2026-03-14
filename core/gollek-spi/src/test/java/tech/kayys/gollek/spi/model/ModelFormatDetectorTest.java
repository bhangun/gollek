package tech.kayys.gollek.spi.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ModelFormatDetector} — magic-byte detection and
 * extension fallback for GGUF and SafeTensors files.
 */
class ModelFormatDetectorTest {

    @TempDir
    Path tempDir;

    // ── GGUF magic-byte detection ──────────────────────────────────────────

    @Test
    void detectGgufByMagicBytes() throws IOException {
        Path gguf = tempDir.resolve("test-model.gguf");
        // GGUF magic: 0x47475546 = 'G','G','U','F' in little-endian uint32
        byte[] header = new byte[16];
        ByteBuffer.wrap(header, 0, 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putInt(0x46554747); // ASCII "GGUF" in LE
        Files.write(gguf, header);

        Optional<ModelFormat> result = ModelFormatDetector.detect(gguf);

        assertTrue(result.isPresent(), "Should detect GGUF by magic bytes");
        assertEquals(ModelFormat.GGUF, result.get());
    }

    @Test
    void isGgufReturnsTrueForGgufFile() throws IOException {
        Path gguf = tempDir.resolve("model.gguf");
        byte[] header = new byte[16];
        ByteBuffer.wrap(header, 0, 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putInt(0x46554747);
        Files.write(gguf, header);

        assertTrue(ModelFormatDetector.isGguf(gguf));
        assertFalse(ModelFormatDetector.isSafeTensors(gguf));
    }

    // ── SafeTensors magic-byte detection ────────────────────────────────────

    @Test
    void detectSafeTensorsByMagicBytes() throws IOException {
        Path st = tempDir.resolve("model.safetensors");
        // SafeTensors: first 8 bytes = uint64 LE (header length), byte 9 = '{'
        byte[] data = new byte[32];
        ByteBuffer.wrap(data, 0, 8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(64L); // header length = 64 bytes (within valid range)
        data[8] = '{'; // JSON header starts with '{'
        Files.write(st, data);

        Optional<ModelFormat> result = ModelFormatDetector.detect(st);

        assertTrue(result.isPresent(), "Should detect SafeTensors by magic bytes");
        assertEquals(ModelFormat.SAFETENSORS, result.get());
    }

    @Test
    void isSafeTensorsReturnsTrueForSafeTensorsFile() throws IOException {
        Path st = tempDir.resolve("weights.safetensors");
        byte[] data = new byte[32];
        ByteBuffer.wrap(data, 0, 8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(128L);
        data[8] = '{';
        Files.write(st, data);

        assertTrue(ModelFormatDetector.isSafeTensors(st));
        assertFalse(ModelFormatDetector.isGguf(st));
    }

    // ── Extension-only detection ────────────────────────────────────────────

    @Test
    void detectByExtensionGguf() {
        assertEquals(Optional.of(ModelFormat.GGUF),
                ModelFormatDetector.detectByExtension("llama3.gguf"));
        assertEquals(Optional.of(ModelFormat.GGUF),
                ModelFormatDetector.detectByExtension("MODEL.GGUF"));
    }

    @Test
    void detectByExtensionSafeTensors() {
        assertEquals(Optional.of(ModelFormat.SAFETENSORS),
                ModelFormatDetector.detectByExtension("model.safetensors"));
        assertEquals(Optional.of(ModelFormat.SAFETENSORS),
                ModelFormatDetector.detectByExtension("weights.safetensor"));
    }

    @Test
    void detectByExtensionOnnx() {
        assertEquals(Optional.of(ModelFormat.ONNX),
                ModelFormatDetector.detectByExtension("model.onnx"));
    }

    @Test
    void detectByExtensionTfLite() {
        assertEquals(Optional.of(ModelFormat.LITERT),
                ModelFormatDetector.detectByExtension("model.tflite"));
    }

    @Test
    void detectByExtensionUnknown() {
        assertTrue(ModelFormatDetector.detectByExtension("model.bin").isEmpty());
        assertTrue(ModelFormatDetector.detectByExtension("model.pt").isEmpty());
        assertTrue(ModelFormatDetector.detectByExtension("readme.txt").isEmpty());
    }

    @Test
    void detectByExtensionNullAndBlank() {
        assertTrue(ModelFormatDetector.detectByExtension(null).isEmpty());
        assertTrue(ModelFormatDetector.detectByExtension("").isEmpty());
        assertTrue(ModelFormatDetector.detectByExtension("   ").isEmpty());
    }

    // ── Extension fallback when magic fails ─────────────────────────────────

    @Test
    void detectFallsBackToExtensionWhenMagicFails() throws IOException {
        // File with random content but .gguf extension
        Path gguf = tempDir.resolve("random.gguf");
        Files.writeString(gguf, "This is not a real GGUF file but has the extension");

        Optional<ModelFormat> result = ModelFormatDetector.detect(gguf);

        assertTrue(result.isPresent(), "Should fall back to extension detection");
        assertEquals(ModelFormat.GGUF, result.get());
    }

    // ── Edge cases ─────────────────────────────────────────────────────────

    @Test
    void detectReturnsEmptyForNull() {
        assertTrue(ModelFormatDetector.detect(null).isEmpty());
    }

    @Test
    void detectReturnsEmptyForNonexistentFile() {
        assertTrue(ModelFormatDetector.detect(Path.of("/nonexistent/model.bin")).isEmpty());
    }

    @Test
    void detectReturnsEmptyForDirectory() throws IOException {
        Path dir = tempDir.resolve("model-dir");
        Files.createDirectories(dir);

        assertTrue(ModelFormatDetector.detect(dir).isEmpty());
    }

    @Test
    void detectRejectsInvalidSafeTensorsHeader() throws IOException {
        Path st = tempDir.resolve("fake.bin");
        // Header length > max (256 MiB) — should be rejected
        byte[] data = new byte[32];
        ByteBuffer.wrap(data, 0, 8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(300L * 1024 * 1024); // 300 MiB — over limit
        data[8] = '{';
        Files.write(st, data);

        // Should not detect as SafeTensors due to excessive header length
        Optional<ModelFormat> result = ModelFormatDetector.detect(st);
        assertFalse(result.isPresent() && result.get() == ModelFormat.SAFETENSORS,
                "Should reject SafeTensors with header > 256 MiB");
    }
}
