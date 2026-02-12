package tech.kayys.golek.spi.storage;

import io.smallrye.mutiny.Uni;

/**
 * Service for storing model artifacts.
 */
public interface ModelStorageService {

    /**
     * Upload model data to storage.
     * 
     * @return URI of the stored model
     */
    Uni<String> uploadModel(String tenantId, String modelId, String version, byte[] data);

    /**
     * API-key-first alias for {@link #uploadModel(String, String, String, byte[])}.
     */
    default Uni<String> uploadModelByApiKey(String apiKey, String modelId, String version, byte[] data) {
        return uploadModel(apiKey, modelId, version, data);
    }

    /**
     * Download model data from storage.
     */
    Uni<byte[]> downloadModel(String storageUri);

    /**
     * Delete model data from storage.
     */
    Uni<Void> deleteModel(String storageUri);
}
