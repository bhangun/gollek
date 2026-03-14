package tech.kayys.gollek.provider.core.registry;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.registry.LocalModelRegistry;
import tech.kayys.gollek.spi.registry.LocalModelRegistry.ModelEntry;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link DefaultLocalModelRegistry} — the CDI implementation
 * of {@link LocalModelRegistry}.
 *
 * <p>These are plain unit tests (no CDI container needed).
 */
class DefaultLocalModelRegistryTest {

    @TempDir
    Path tempDir;

    private DefaultLocalModelRegistry registry;

    @BeforeEach
    void setUp() {
        registry = new DefaultLocalModelRegistry();
    }

    // ── Explicit registration ──────────────────────────────────────────────

    @Test
    void registerAndResolveByExactId() throws IOException {
        Path gguf = createGgufFile("llama3.gguf");

        ModelEntry entry = registry.register("llama3", gguf, ModelFormat.GGUF);

        assertEquals("llama3", entry.modelId());
        assertEquals(ModelFormat.GGUF, entry.format());
        assertEquals(gguf.toAbsolutePath().normalize(), entry.physicalPath());

        Optional<ModelEntry> resolved = registry.resolve("llama3");
        assertTrue(resolved.isPresent());
        assertEquals("llama3", resolved.get().modelId());
    }

    @Test
    void registerOverwritesPreviousEntry() throws IOException {
        Path v1 = createGgufFile("model-v1.gguf");
        Path v2 = createGgufFile("model-v2.gguf");

        registry.register("my-model", v1, ModelFormat.GGUF);
        registry.register("my-model", v2, ModelFormat.GGUF);

        ModelEntry entry = registry.resolve("my-model").orElseThrow();
        assertEquals(v2.toAbsolutePath().normalize(), entry.physicalPath());
    }

    // ── Alias resolution ───────────────────────────────────────────────────

    @Test
    void resolveViaAlias() throws IOException {
        Path gguf = createGgufFile("tinyllama-1.1b.gguf");
        registry.register("tinyllama-1.1b", gguf, ModelFormat.GGUF);
        registry.registerAlias("tinyllama", "tinyllama-1.1b");

        Optional<ModelEntry> resolved = registry.resolve("tinyllama");
        assertTrue(resolved.isPresent());
        assertEquals("tinyllama-1.1b", resolved.get().modelId());
    }

    // ── Absolute path resolution ───────────────────────────────────────────

    @Test
    void resolveFromAbsolutePath() throws IOException {
        Path gguf = createGgufFile("direct.gguf");

        Optional<ModelEntry> resolved = registry.resolve(gguf.toString());
        assertTrue(resolved.isPresent());
        assertEquals(ModelFormat.GGUF, resolved.get().format());
    }

    // ── Directory scanning ─────────────────────────────────────────────────

    @Test
    void scanRootDiscoverModels() throws IOException {
        Path ggufDir = tempDir.resolve("gguf-models");
        Files.createDirectories(ggufDir);
        createGgufFile(ggufDir, "model-a.gguf");
        createGgufFile(ggufDir, "model-b.gguf");

        registry.addScanRoots(ggufDir);
        registry.refresh();

        List<ModelEntry> all = registry.listAll(ModelFormat.GGUF);
        assertTrue(all.size() >= 2, "Should discover at least 2 GGUF models, found: " + all.size());
    }

    @Test
    void scanRootIgnoresNonModelFiles() throws IOException {
        Path dir = tempDir.resolve("mixed");
        Files.createDirectories(dir);
        createGgufFile(dir, "model.gguf");
        Files.writeString(dir.resolve("readme.txt"), "Not a model");
        Files.writeString(dir.resolve("data.json"), "{}");

        registry.addScanRoots(dir);
        registry.refresh();

        List<ModelEntry> all = registry.listAll(null);
        // Should find only the GGUF file
        assertEquals(1, all.stream()
                .filter(e -> e.physicalPath().getFileName().toString().equals("model.gguf"))
                .count());
    }

    @Test
    void addScanRootsSkipsNonDirectories() throws IOException {
        Path file = tempDir.resolve("not-a-directory.txt");
        Files.writeString(file, "hello");

        // Should not throw
        registry.addScanRoots(file);
        registry.refresh();
        assertTrue(registry.listAll(null).isEmpty());
    }

    // ── Fuzzy match ────────────────────────────────────────────────────────

    @Test
    void fuzzyMatchByStem() throws IOException {
        Path gguf = createGgufFile("my-fancy-model.gguf");
        registry.register(gguf.toAbsolutePath().normalize().toString(), gguf, ModelFormat.GGUF);

        // Fuzzy match should find by stem (without extension)
        Optional<ModelEntry> resolved = registry.resolve("my-fancy-model");
        assertTrue(resolved.isPresent(),
                "Fuzzy match should find 'my-fancy-model' without .gguf extension");
    }

    @Test
    void fuzzyMatchIsCaseInsensitive() throws IOException {
        Path gguf = createGgufFile("TinyLlama.gguf");
        registry.register(gguf.toAbsolutePath().normalize().toString(), gguf, ModelFormat.GGUF);

        Optional<ModelEntry> resolved = registry.resolve("tinyllama");
        assertTrue(resolved.isPresent(),
                "Fuzzy match should be case-insensitive");
    }

    // ── listAll filtering ──────────────────────────────────────────────────

    @Test
    void listAllFiltersByFormat() throws IOException {
        Path gguf = createGgufFile("model.gguf");
        Path st = createSafeTensorsFile("model.safetensors");

        registry.register("gguf-model", gguf, ModelFormat.GGUF);
        registry.register("st-model", st, ModelFormat.SAFETENSORS);

        List<ModelEntry> ggufOnly = registry.listAll(ModelFormat.GGUF);
        assertEquals(1, ggufOnly.size());
        assertEquals("gguf-model", ggufOnly.get(0).modelId());

        List<ModelEntry> stOnly = registry.listAll(ModelFormat.SAFETENSORS);
        assertEquals(1, stOnly.size());
        assertEquals("st-model", stOnly.get(0).modelId());

        List<ModelEntry> all = registry.listAll(null);
        assertEquals(2, all.size());
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    @Test
    void clearRemovesAllEntries() throws IOException {
        Path gguf = createGgufFile("model.gguf");
        registry.register("test", gguf, ModelFormat.GGUF);
        registry.registerAlias("alias", "test");

        assertFalse(registry.listAll(null).isEmpty());

        registry.clear();

        assertTrue(registry.listAll(null).isEmpty());
        assertTrue(registry.resolve("test").isEmpty());
        assertTrue(registry.resolve("alias").isEmpty());
    }

    @Test
    void resolveReturnsEmptyForUnknownModel() {
        assertTrue(registry.resolve("nonexistent-model").isEmpty());
    }

    @Test
    void resolveReturnsEmptyForNull() {
        assertTrue(registry.resolve(null).isEmpty());
        assertTrue(registry.resolve("").isEmpty());
    }

    // ── SafeTensors discovery ──────────────────────────────────────────────

    @Test
    void scanDiscoversSafeTensorsFiles() throws IOException {
        Path stDir = tempDir.resolve("safetensors-models");
        Files.createDirectories(stDir);
        createSafeTensorsFile(stDir, "bert.safetensors");

        registry.addScanRoots(stDir);
        registry.refresh();

        List<ModelEntry> stModels = registry.listAll(ModelFormat.SAFETENSORS);
        assertTrue(stModels.size() >= 1,
                "Should discover SafeTensors model, found: " + stModels.size());
    }

    // ── ModelEntry displayName ──────────────────────────────────────────────

    @Test
    void displayNameForGgufIsFileName() throws IOException {
        Path gguf = createGgufFile("llama.gguf");
        ModelEntry entry = registry.register("test", gguf, ModelFormat.GGUF);

        assertEquals("llama.gguf", entry.displayName());
    }

    @Test
    void displayNameForSafeTensorsIsParentDir() throws IOException {
        Path modelDir = tempDir.resolve("bert-base");
        Files.createDirectories(modelDir);
        Path st = createSafeTensorsFile(modelDir, "model.safetensors");

        ModelEntry entry = registry.register("test", st, ModelFormat.SAFETENSORS);

        assertEquals("bert-base", entry.displayName());
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private Path createGgufFile(String name) throws IOException {
        return createGgufFile(tempDir, name);
    }

    private Path createGgufFile(Path dir, String name) throws IOException {
        Path file = dir.resolve(name);
        byte[] header = new byte[16];
        ByteBuffer.wrap(header, 0, 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putInt(0x46554747); // GGUF magic
        Files.write(file, header);
        return file;
    }

    private Path createSafeTensorsFile(String name) throws IOException {
        return createSafeTensorsFile(tempDir, name);
    }

    private Path createSafeTensorsFile(Path dir, String name) throws IOException {
        Path file = dir.resolve(name);
        byte[] data = new byte[32];
        ByteBuffer.wrap(data, 0, 8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(64L);
        data[8] = '{';
        Files.write(file, data);
        return file;
    }
}
