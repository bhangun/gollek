package tech.kayys.golek.model.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.Pageable;

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
        String tenantId = "tenant-1";
        String modelId = "model-1";
        ModelManifest manifest = createManifest(modelId, tenantId);

        repository.save(manifest).await().indefinitely();

        ModelManifest found = repository.findById(modelId, tenantId).await().indefinitely();
        assertNotNull(found);
        assertEquals(modelId, found.modelId());
        assertEquals(tenantId, found.tenantId());
    }

    @Test
    void testListWithPagination() {
        String tenantId = "tenant-1";
        repository.save(createManifest("m1", tenantId)).await().indefinitely();
        repository.save(createManifest("m2", tenantId)).await().indefinitely();
        repository.save(createManifest("m3", tenantId)).await().indefinitely();

        List<ModelManifest> page1 = repository.list(tenantId, Pageable.of(0, 2)).await().indefinitely();
        assertEquals(2, page1.size());

        List<ModelManifest> page2 = repository.list(tenantId, Pageable.of(1, 2)).await().indefinitely();
        assertEquals(1, page2.size());
    }

    @Test
    void testDelete() {
        String tenantId = "tenant-1";
        String modelId = "model-1";
        repository.save(createManifest(modelId, tenantId)).await().indefinitely();

        repository.delete(modelId, tenantId).await().indefinitely();

        ModelManifest found = repository.findById(modelId, tenantId).await().indefinitely();
        assertNull(found);
    }

    private ModelManifest createManifest(String modelId, String tenantId) {
        return new ModelManifest(
                modelId,
                modelId + " Name",
                "1.0.0",
                tenantId,
                Map.of(),
                List.of(),
                null,
                Map.of(),
                Instant.now(),
                Instant.now());
    }
}
