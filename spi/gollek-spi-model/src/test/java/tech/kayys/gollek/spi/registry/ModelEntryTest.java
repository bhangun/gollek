package tech.kayys.gollek.spi.registry;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.core.model.ModelFormat;

import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelEntryTest {

    @Test
    void testModelEntryCreation() {
        ModelEntry entry = new ModelEntry(
                "model-1",
                "Model 1",
                ModelFormat.GGUF,
                Path.of("/tmp/model.gguf"),
                1024L,
                "gguf-provider",
                Map.of("version", "v1"),
                Instant.now());

        assertEquals("model-1", entry.modelId());
        assertEquals("Model 1", entry.name());
        assertEquals(ModelFormat.GGUF, entry.format());
        assertEquals(Path.of("/tmp/model.gguf"), entry.physicalPath());
        assertEquals(1024L, entry.sizeBytes());
        assertEquals("gguf-provider", entry.provider());
        assertEquals("v1", entry.metadata().get("version"));
        assertNotNull(entry.registeredAt());
    }

    @Test
    void testModelEntryBuilder() {
        ModelEntry entry = ModelEntry.builder()
                .modelId("model-2")
                .name("Model 2")
                .format(ModelFormat.ONNX)
                .physicalPath(Path.of("/tmp/model.onnx"))
                .sizeBytes(2048L)
                .provider("onnx-provider")
                .metadata(Map.of("type", "chat"))
                .registeredAt(Instant.EPOCH)
                .build();

        assertEquals("model-2", entry.modelId());
        assertEquals(ModelFormat.ONNX, entry.format());
        assertEquals(Instant.EPOCH, entry.registeredAt());
    }

    @Test
    void testAliasMethods() {
        ModelEntry entry = ModelEntry.builder()
                .physicalPath(Path.of("/path/to/model"))
                .build();

        assertEquals(entry.physicalPath(), entry.path());
    }
}
