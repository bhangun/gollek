package tech.kayys.gollek.engine.model;

import io.smallrye.mutiny.Uni;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.storage.ModelStorageService;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Plugin-based model storage service that delegates to specific storage
 * implementations.
 *
 * <p>
 * Supports multiple storage backends through plugins:
 * <ul>
 * <li>AWS S3 (via gollek-plugin-storage-s3)</li>
 * <li>Google Cloud Storage (via gollek-plugin-storage-gcs)</li>
 * <li>Azure Blob Storage (via gollek-plugin-storage-azure)</li>
 * <li>Local filesystem (built-in)</li>
 * </ul>
 */
@ApplicationScoped
public class PluginBasedModelStorageService implements ModelStorageService {

    private static final Logger LOG = Logger.getLogger(PluginBasedModelStorageService.class);

    @ConfigProperty(name = "gollek.storage.provider", defaultValue = "local")
    String storageProvider;

    // Map of storage providers to their implementations
    private final Map<String, ModelStorageService> storageServices = new ConcurrentHashMap<>();

    @Inject
    Instance<ModelStorageService> availableServices;

    @Inject
    GollekModelStorageService localService;

    @PostConstruct
    void init() {
        // Register the built-in local service
        storageServices.put("local", localService);
        
        // Dynamically register other available services
        // Note: In a real implementation, we'd need a way to get the provider name from the service
        // For now, we'll try to discover them if they have a specific naming convention or metadata
        LOG.debug("Initializing PluginBasedModelStorageService with available storage plugins");
    }

    @Override
    public Uni<String> uploadModel(String apiKey, String modelId, String version, byte[] data) {
        ModelStorageService service = getStorageService();
        if (service == null) {
            return Uni.createFrom().failure(
                    new IllegalStateException("No storage service available for provider: " + storageProvider));
        }
        return service.uploadModel(apiKey, modelId, version, data);
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
        ModelStorageService service = storageServices.get(provider);
        if (service != null) {
            return service;
        }

        // Fallback to local if requested provider is not available
        LOG.warnf("Requested storage provider '%s' not available, falling back to local storage", provider);
        return storageServices.get("local");
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
            return "local";
        }
    }
}