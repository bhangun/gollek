package tech.kayys.gollek.plugin.storage.gcs;

import io.smallrye.mutiny.Uni;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;
import com.google.cloud.storage.BlobId;
import com.google.cloud.storage.BlobInfo;
import tech.kayys.gollek.spi.storage.ModelStorageService;

import jakarta.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class GCSModelStorageService implements ModelStorageService {

    private Storage storage;
    private String bucketName;
    private String pathPrefix;

    // Initialize with configuration
    public void initialize(String bucketName, String pathPrefix) {
        this.bucketName = bucketName;
        this.pathPrefix = pathPrefix != null ? pathPrefix : "models/";

        // Use default storage options; project ID is typically handled by environment
        // or ADC
        this.storage = StorageOptions.getDefaultInstance().getService();
    }

    @Override
    public Uni<String> uploadModel(String apiKey, String modelId, String version, byte[] data) {
        return Uni.createFrom().item(() -> {
            try {
                String blobName = String.format("%s%s/%s/%s", pathPrefix, apiKey, modelId, version);

                BlobId blobId = BlobId.of(bucketName, blobName);
                BlobInfo blobInfo = BlobInfo.newBuilder(blobId).build();

                storage.create(blobInfo, data);

                return String.format("gs://%s/%s", bucketName, blobName);
            } catch (Exception e) {
                throw new RuntimeException("Failed to upload model to GCS", e);
            }
        });
    }

    @Override
    public Uni<byte[]> downloadModel(String storageUri) {
        return Uni.createFrom().item(() -> {
            try {
                // Extract blob name from GCS URI (format: gs://bucket/blob-name)
                String blobName = storageUri.substring(("gs://" + bucketName + "/").length());

                BlobId blobId = BlobId.of(bucketName, blobName);

                return storage.readAllBytes(blobId);
            } catch (Exception e) {
                throw new RuntimeException("Failed to download model from GCS", e);
            }
        });
    }

    @Override
    public Uni<Void> deleteModel(String storageUri) {
        return Uni.createFrom().item(() -> {
            try {
                // Extract blob name from GCS URI (format: gs://bucket/blob-name)
                String blobName = storageUri.substring(("gs://" + bucketName + "/").length());

                BlobId blobId = BlobId.of(bucketName, blobName);

                storage.delete(blobId);
                return null;
            } catch (Exception e) {
                throw new RuntimeException("Failed to delete model from GCS", e);
            }
        });
    }
}