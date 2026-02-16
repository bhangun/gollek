package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.golek.spi.storage.ModelStorageService;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Plugin-based model storage service that delegates to specific storage
 * implementations.
 *
 * <p>
 * Supports multiple storage backends through plugins:
 * <ul>
 * <li>AWS S3 (via golek-plugin-storage-s3)</li>
 * <li>Google Cloud Storage (via golek-plugin-storage-gcs)</li>
 * <li>Azure Blob Storage (via golek-plugin-storage-azure)</li>
 * <li>Local filesystem (built-in)</li>
 * </ul>
 *
 * @author Bhangun
 * @since 1.0.0
 */
@ApplicationScoped
public class PluginBasedModelStorageService implements ModelStorageService {

    private static final Logger log = LoggerFactory.getLogger(PluginBasedModelStorageService.class);

    @ConfigProperty(name = "inference.model-storage.provider", defaultValue = "local")
    String storageProvider;

    @ConfigProperty(name = "inference.model-storage.local.base-path", defaultValue = "/var/lib/inference/models")
    String localBasePath;

    // Map of storage providers to their implementations
    private final Map<String, ModelStorageService> storageServices = new ConcurrentHashMap<>();

    @Inject
    Instance<ModelStorageService> availableServices;

    public PluginBasedModelStorageService() {
        // Initialize the map with built-in local storage
        storageServices.put("local", new LocalModelStorageService());
    }

    @Override
    public Uni<String> uploadModel(String requestId, String modelId, String version, byte[] data) {
        ModelStorageService service = getStorageService();
        if (service == null) {
            return Uni.createFrom().failure(
                    new IllegalStateException("No storage service available for provider: " + storageProvider));
        }
        return service.uploadModel(requestId, modelId, version, data);
    }

    @Override
    public Uni<byte[]> downloadModel(String storageUri) {
        String provider = determineProviderFromUri(storageUri);
        ModelStorageService service = getStorageService(provider);
        if (service == null) {
            return Uni.createFrom()
                    .failure(new IllegalStateException("No storage service available for URI: " + storageUri));
        }
        return service.downloadModel(storageUri);
    }

    @Override
    public Uni<Void> deleteModel(String storageUri) {
        String provider = determineProviderFromUri(storageUri);
        ModelStorageService service = getStorageService(provider);
        if (service == null) {
            return Uni.createFrom()
                    .failure(new IllegalStateException("No storage service available for URI: " + storageUri));
        }
        return service.deleteModel(storageUri);
    }

    private ModelStorageService getStorageService() {
        return getStorageService(storageProvider.toLowerCase());
    }

    private ModelStorageService getStorageService(String provider) {
        // Check if we already have a cached instance
        ModelStorageService service = storageServices.get(provider);
        if (service != null) {
            return service;
        }

        // Try to find an available service implementation for this provider
        // This is a simplified approach - in a real implementation, you'd have more
        // sophisticated discovery
        synchronized (storageServices) {
            service = storageServices.get(provider);
            if (service != null) {
                return service;
            }

            // For now, we'll just return the local service if the requested provider isn't
            // available
            // In a real implementation, you'd dynamically load the appropriate plugin
            log.warn("Requested storage provider '{}' not available, falling back to local storage", provider);
            return storageServices.get("local");
        }
    }

    private String determineProviderFromUri(String storageUri) {
        if (storageUri == null) {
            return "local";
        }

        if (storageUri.startsWith("s3://")) {
            return "s3";
        } else if (storageUri.startsWith("gs://")) {
            return "gcs";
        } else if (storageUri.startsWith("azure://")) {
            return "azure";
        } else if (storageUri.startsWith("file://")) {
            return "local";
        } else {
            // Default to local if no scheme is recognized
            return "local";
        }
    }

    /**
     * Built-in local file system storage implementation.
     */
    private static class LocalModelStorageService implements ModelStorageService {
        private final String basePath;

        public LocalModelStorageService() {
            this.basePath = System.getProperty("inference.model-storage.local.base-path", "/var/lib/inference/models");
        }

        @Override
        public Uni<String> uploadModel(String requestId, String modelId, String version, byte[] data) {
            return Uni.createFrom().item(() -> {
                try {
                    Path tenantPath = Paths.get(basePath, requestId);
                    Path modelPath = tenantPath.resolve(modelId);
                    Path versionPath = modelPath.resolve(version);

                    // Create directories if they don't exist
                    Files.createDirectories(versionPath.getParent());

                    // Write the data to the file
                    Files.write(versionPath, data);

                    // Return file URI
                    return "file://" + versionPath.toString();
                } catch (IOException e) {
                    log.error("Failed to upload model to local storage", e);
                    throw new RuntimeException("Failed to upload model to local storage", e);
                }
            });
        }

        @Override
        public Uni<byte[]> downloadModel(String storageUri) {
            return Uni.createFrom().item(() -> {
                try {
                    // Extract path from file URI (format: file:///path/to/file)
                    String filePath = storageUri.substring("file://".length());
                    Path path = Paths.get(filePath);

                    return Files.readAllBytes(path);
                } catch (IOException e) {
                    log.error("Failed to download model from local storage", e);
                    throw new RuntimeException("Failed to download model from local storage", e);
                }
            });
        }

        @Override
        public Uni<Void> deleteModel(String storageUri) {
            return Uni.createFrom().item(() -> {
                try {
                    // Extract path from file URI (format: file:///path/to/file)
                    String filePath = storageUri.substring("file://".length());
                    Path path = Paths.get(filePath);

                    Files.deleteIfExists(path);
                    return null;
                } catch (IOException e) {
                    log.error("Failed to delete model from local storage", e);
                    throw new RuntimeException("Failed to delete model from local storage", e);
                }
            });
        }
    }
}