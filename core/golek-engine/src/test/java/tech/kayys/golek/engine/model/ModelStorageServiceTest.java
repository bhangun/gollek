package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.junit.jupiter.api.Assertions.*;

class ModelStorageServiceTest {

    @Mock
    private tech.kayys.golek.spi.storage.ModelStorageService mockStorageService;

    private GolekModelStorageService golekModelStorageService;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);

        // Create an instance of the actual service
        golekModelStorageService = new GolekModelStorageService();
        // Manually inject dependencies for the test (Quarkus would do this in real app)
        // For ExecutorService, we can use a direct one for test
        golekModelStorageService.executorService = java.util.concurrent.Executors.newSingleThreadExecutor();

        // Set some test configuration values
        golekModelStorageService.storageProvider = "local";
        golekModelStorageService.localBasePath = "/tmp/test-models-golek"; // Use a different path for isolation
    }

    @Test
    void testUploadModelWithValidInputs() {
        String requestId = "test-tenant";
        String modelId = "test-model";
        String version = "v1.0.0";
        byte[] modelData = "test model data".getBytes();

        // Since we're using local storage in test, this should work
        Uni<String> result = golekModelStorageService.uploadModel(requestId, modelId, version, modelData);

        // Wait for the async operation to complete
        String storageUri = result.await().indefinitely();

        assertNotNull(storageUri);
        assertTrue(storageUri.startsWith("file://"));
        assertTrue(storageUri.contains("/tmp/test-models/" + requestId + "/" + modelId + "/" + version));
    }

    @Test
    void testUploadModelWithNullRequestId() {
        String requestId = null;
        String modelId = "test-model";
        String version = "v1.0.0";
        byte[] modelData = "test model data".getBytes();

        Uni<String> result = golekModelStorageService.uploadModel(requestId, modelId, version, modelData);

        Exception exception = assertThrows(RuntimeException.class, () -> {
            result.await().indefinitely();
        });

        assertTrue(
                exception instanceof IllegalArgumentException
                        || exception.getCause() instanceof IllegalArgumentException,
                "Expected IllegalArgumentException but got: " + exception);
    }

    @Test
    void testUploadModelWithEmptyModelId() {
        String requestId = "test-tenant";
        String modelId = "";
        String version = "v1.0.0";
        byte[] modelData = "test model data".getBytes();

        Uni<String> result = golekModelStorageService.uploadModel(requestId, modelId, version, modelData);

        Exception exception = assertThrows(RuntimeException.class, () -> {
            result.await().indefinitely();
        });

        assertTrue(
                exception instanceof IllegalArgumentException
                        || exception.getCause() instanceof IllegalArgumentException,
                "Expected IllegalArgumentException but got: " + exception);
    }

    @Test
    void testUploadModelWithNullModelData() {
        String requestId = "test-tenant";
        String modelId = "test-model";
        String version = "v1.0.0";
        byte[] modelData = null;

        Uni<String> result = golekModelStorageService.uploadModel(requestId, modelId, version, modelData);

        Exception exception = assertThrows(RuntimeException.class, () -> {
            result.await().indefinitely();
        });

        assertTrue(
                exception instanceof IllegalArgumentException
                        || exception.getCause() instanceof IllegalArgumentException,
                "Expected IllegalArgumentException but got: " + exception);
    }

    @Test
    void testDownloadModelWithValidUri() {
        // First upload a model to get a valid URI
        String requestId = "test-tenant";
        String modelId = "test-model";
        String version = "v1.0.0";
        byte[] originalData = "test model data for download".getBytes();

        String storageUri = golekModelStorageService.uploadModel(requestId, modelId, version, originalData)
                .await().indefinitely();

        // Now download the model
        Uni<byte[]> result = golekModelStorageService.downloadModel(storageUri);
        byte[] downloadedData = result.await().indefinitely();

        assertArrayEquals(originalData, downloadedData);
    }

    @Test
    void testDownloadModelWithInvalidUri() {
        String invalidUri = "invalid-uri-format";

        Uni<byte[]> result = golekModelStorageService.downloadModel(invalidUri);

        Exception exception = assertThrows(RuntimeException.class, () -> {
            result.await().indefinitely();
        });

        assertTrue(
                exception instanceof IllegalArgumentException
                        || exception.getCause() instanceof IllegalArgumentException,
                "Expected IllegalArgumentException but got: " + exception);
    }

    @Test
    void testModelExistsWithExistingModel() {
        // First upload a model to get a valid URI
        String requestId = "test-tenant";
        String modelId = "test-model";
        String version = "v1.0.0";
        byte[] modelData = "test model data for exists check".getBytes();

        String storageUri = golekModelStorageService.uploadModel(requestId, modelId, version, modelData)
                .await().indefinitely();

        // Check if the model exists
        Uni<Boolean> result = golekModelStorageService.modelExists(storageUri);
        Boolean exists = result.await().indefinitely();

        assertTrue(exists);
    }

    @Test
    void testDeleteModel() {
        // First upload a model to get a valid URI
        String requestId = "test-tenant";
        String modelId = "test-model";
        String version = "v1.0.0";
        byte[] modelData = "test model data for deletion".getBytes();

        String storageUri = golekModelStorageService.uploadModel(requestId, modelId, version, modelData)
                .await().indefinitely();

        // Verify the model exists before deletion
        Boolean existsBefore = golekModelStorageService.modelExists(storageUri).await().indefinitely();
        assertTrue(existsBefore);

        // Delete the model
        Uni<Void> deleteResult = golekModelStorageService.deleteModel(storageUri);
        deleteResult.await().indefinitely(); // Wait for deletion to complete

        // Verify the model no longer exists
        Boolean existsAfter = golekModelStorageService.modelExists(storageUri).await().indefinitely();
        assertFalse(existsAfter);
    }

}