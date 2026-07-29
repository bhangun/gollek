package tech.kayys.gollek.spi.model;

import tech.kayys.gollek.spi.spec.*;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;

import io.smallrye.mutiny.Uni;

/**
 * Service Provider Interface (SPI) for model artifact storage.
 * 
 * <p>
 * This interface abstracts the underlying storage mechanism used to persist
 * machine learning model files (e.g., GGUF, SafeTensors, ONNX). Implementations
 * can provide support for local filesystems, cloud storage (S3, GCS, Azure),
 * or specialized model repositories.
 * </p>
 * 
 * <p>
 * All operations are asynchronous and return {@link Uni} for reactive
 * processing.
 * </p>
 */
public interface ModelStorageService {

    /**
     * Uploads model data to the storage backend.
     * 
     * @param apiKey  The API key or Tenant ID used for access control and logical
     *                isolation.
     * @param modelId The unique identifier of the model.
     * @param version The version string of the model (e.g., "1.0.0" or "latest").
     * @param data    The raw binary content of the model file.
     * @return A {@link Uni} that emits the storage URI (e.g.,
     *         {@code file:///path/to/model.bin}
     *         or {@code s3://bucket/model.bin}) upon successful upload.
     * @throws IllegalArgumentException if any parameter is null or blank.
     */
    Uni<String> uploadModel(String apiKey, String modelId, String version, byte[] data);

    /**
     * An alias for {@link #uploadModel(String, String, String, byte[])} that
     * emphasizes
     * the use of an API key for isolation.
     * 
     * @param apiKey  The API key identifying the owner of the model.
     * @param modelId The unique identifier of the model.
     * @param version The version of the model.
     * @param data    The model binary data.
     * @return A {@link Uni} emitting the storage URI.
     */
    default Uni<String> uploadModelByApiKey(String apiKey, String modelId, String version, byte[] data) {
        return uploadModel(apiKey, modelId, version, data);
    }

    /**
     * Downloads model data from the given storage URI.
     * 
     * @param storageUri The full URI of the model as returned by
     *                   {@link #uploadModel}.
     * @return A {@link Uni} that emits the raw binary data of the model.
     * @throws IllegalArgumentException if the storage URI is invalid or unsupported
     *                                  by the implementation.
     */
    Uni<byte[]> downloadModel(String storageUri);

    /**
     * Deletes a model from the storage backend.
     * 
     * @param storageUri The full URI of the model to be deleted.
     * @return A {@link Uni} that emits void when the deletion is complete.
     * @throws IllegalArgumentException if the storage URI is invalid.
     */
    Uni<Void> deleteModel(String storageUri);
}
