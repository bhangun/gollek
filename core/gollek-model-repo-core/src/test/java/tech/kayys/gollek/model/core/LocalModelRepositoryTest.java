package tech.kayys.gollek.model.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.Pageable;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class LocalModelRepositoryTest {

    @TempDir
    Path tempDir;

    private LocalModelRepository repository;

    @BeforeEach
    void setUp() {
        repository = new LocalModelRepository(tempDir.toString());
    }

    @Test
    void testSaveAndFind() {
        String path = "tenant-1";
        String modelId = "model-1";
        ModelManifest manifest = createManifest(modelId, path);

        repository.save(manifest).await().indefinitely();

        ModelManifest found = repository.findById(modelId, path).await().indefinitely();
        assertNotNull(found);
        assertEquals(modelId, found.modelId());
        assertEquals(path, found.path());
    }

    @Test
    void testListWithPagination() {
        String path = "tenant-1";
        repository.save(createManifest("m1", path)).await().indefinitely();
        repository.save(createManifest("m2", path)).await().indefinitely();
        repository.save(createManifest("m3", path)).await().indefinitely();

        List<ModelManifest> page1 = repository.list(path, Pageable.of(0, 2)).await().indefinitely();
        assertEquals(2, page1.size());

        List<ModelManifest> page2 = repository.list(path, Pageable.of(1, 2)).await().indefinitely();
        assertEquals(1, page2.size());
    }

    @Test
    void testDelete() {
        String path = "tenant-1";
        String modelId = "model-1";
        repository.save(createManifest(modelId, path)).await().indefinitely();

        repository.delete(modelId, path).await().indefinitely();

        ModelManifest found = repository.findById(modelId, path).await().indefinitely();
        assertNull(found);
    }

    private ModelManifest createManifest(String modelId, String path) {
        return new ModelManifest(
                modelId,
                modelId + " Name",
                "1.0.0",
                path,
                "",
                "",
                Map.of(),
                List.of(),
                null,
                Map.of(),
                Instant.now(),
                Instant.now());
    }
}
