package tech.kayys.golek.plugin.storage.azure;

import io.smallrye.mutiny.Uni;
import com.azure.storage.blob.BlobClient;
import com.azure.storage.blob.BlobContainerClient;
import com.azure.storage.blob.BlobServiceClient;
import com.azure.storage.blob.BlobServiceClientBuilder;
import com.azure.core.util.BinaryData;
import tech.kayys.golek.spi.storage.ModelStorageService;

import jakarta.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class AzureModelStorageService implements ModelStorageService {

    private BlobContainerClient containerClient;
    private String containerName;
    private String pathPrefix;

    // Initialize with configuration
    public void initialize(String connectionString, String containerName, String pathPrefix) {
        this.containerName = containerName;
        this.pathPrefix = pathPrefix != null ? pathPrefix : "models/";

        BlobServiceClient blobServiceClient = new BlobServiceClientBuilder()
                .connectionString(connectionString)
                .buildClient();

        this.containerClient = blobServiceClient.getBlobContainerClient(containerName);
    }

    @Override
    public Uni<String> uploadModel(String requestId, String modelId, String version, byte[] data) {
        return Uni.createFrom().item(() -> {
            try {
                String blobName = String.format("%s%s/%s/%s", pathPrefix, requestId, modelId, version);

                BlobClient blobClient = containerClient.getBlobClient(blobName);

                blobClient.upload(BinaryData.fromBytes(data), true);

                return String.format("azure://%s/%s", containerName, blobName);
            } catch (Exception e) {
                throw new RuntimeException("Failed to upload model to Azure", e);
            }
        });
    }

    @Override
    public Uni<byte[]> downloadModel(String storageUri) {
        return Uni.createFrom().item(() -> {
            try {
                // Extract blob name from Azure URI (format: azure://container/blob-name)
                String blobName = storageUri.substring(("azure://" + containerName + "/").length());

                BlobClient blobClient = containerClient.getBlobClient(blobName);

                return blobClient.downloadContent().toBytes();
            } catch (Exception e) {
                throw new RuntimeException("Failed to download model from Azure", e);
            }
        });
    }

    @Override
    public Uni<Void> deleteModel(String storageUri) {
        return Uni.createFrom().item(() -> {
            try {
                // Extract blob name from Azure URI (format: azure://container/blob-name)
                String blobName = storageUri.substring(("azure://" + containerName + "/").length());

                BlobClient blobClient = containerClient.getBlobClient(blobName);

                blobClient.delete();
                return null;
            } catch (Exception e) {
                throw new RuntimeException("Failed to delete model from Azure", e);
            }
        });
    }
}