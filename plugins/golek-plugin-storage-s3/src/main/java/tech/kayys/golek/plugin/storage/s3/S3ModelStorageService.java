package tech.kayys.golek.plugin.storage.s3;

import io.smallrye.mutiny.Uni;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.*;
import tech.kayys.golek.spi.storage.ModelStorageService;

import jakarta.enterprise.context.ApplicationScoped;
import java.net.URI;

@ApplicationScoped
public class S3ModelStorageService implements ModelStorageService {

    private S3Client s3Client;
    private String bucketName;
    private String region;
    private String pathPrefix;

    // Initialize with configuration
    public void initialize(String accessKeyId, String secretAccessKey, String bucketName, 
                          String region, String endpoint, String pathPrefix) {
        this.bucketName = bucketName;
        this.region = region;
        this.pathPrefix = pathPrefix != null ? pathPrefix : "models/";

        AwsBasicCredentials awsCreds = AwsBasicCredentials.create(accessKeyId, secretAccessKey);
        S3Client.Builder builder = S3Client.builder()
                .credentialsProvider(StaticCredentialsProvider.create(awsCreds))
                .region(Region.of(region));

        if (endpoint != null && !endpoint.trim().isEmpty()) {
            builder.endpointOverride(URI.create(endpoint));
        }

        this.s3Client = builder.build();
    }

    @Override
    public Uni<String> uploadModel(String tenantId, String modelId, String version, byte[] data) {
        return Uni.createFrom().item(() -> {
            try {
                String key = String.format("%s%s/%s/%s", pathPrefix, tenantId, modelId, version);
                
                PutObjectRequest request = PutObjectRequest.builder()
                        .bucket(bucketName)
                        .key(key)
                        .build();

                s3Client.putObject(request, RequestBody.fromBytes(data));

                return String.format("s3://%s/%s", bucketName, key);
            } catch (Exception e) {
                throw new RuntimeException("Failed to upload model to S3", e);
            }
        });
    }

    @Override
    public Uni<byte[]> downloadModel(String storageUri) {
        return Uni.createFrom().item(() -> {
            try {
                // Extract key from S3 URI (format: s3://bucket/key)
                String key = storageUri.substring(("s3://" + bucketName + "/").length());
                
                GetObjectRequest request = GetObjectRequest.builder()
                        .bucket(bucketName)
                        .key(key)
                        .build();

                return s3Client.getObjectAsBytes(request).asByteArray();
            } catch (Exception e) {
                throw new RuntimeException("Failed to download model from S3", e);
            }
        });
    }

    @Override
    public Uni<Void> deleteModel(String storageUri) {
        return Uni.createFrom().item(() -> {
            try {
                // Extract key from S3 URI (format: s3://bucket/key)
                String key = storageUri.substring(("s3://" + bucketName + "/").length());
                
                DeleteObjectRequest request = DeleteObjectRequest.builder()
                        .bucket(bucketName)
                        .key(key)
                        .build();

                s3Client.deleteObject(request);
                return null;
            } catch (Exception e) {
                throw new RuntimeException("Failed to delete model from S3", e);
            }
        });
    }
}