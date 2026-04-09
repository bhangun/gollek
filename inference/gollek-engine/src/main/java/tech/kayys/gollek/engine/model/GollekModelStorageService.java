package tech.kayys.gollek.engine.model;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.storage.ModelStorageService;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.stream.Stream;

/**
 * Local implementation of the model storage service.
 * Stores models in a local directory structure.
 */
@ApplicationScoped
public class GollekModelStorageService implements ModelStorageService {

    private static final Logger LOG = Logger.getLogger(GollekModelStorageService.class);

    @ConfigProperty(name = "gollek.storage.provider", defaultValue = "local")
    String storageProvider;

    @ConfigProperty(name = "gollek.storage.local.base-path", defaultValue = "/tmp/gollek-models")
    String localBasePath;

    @Inject
    ExecutorService executorService;

    @Override
    public Uni<String> uploadModel(String apiKey, String modelId, String version, byte[] data) {
        if (!"local".equalsIgnoreCase(storageProvider)) {
            return Uni.createFrom()
                    .failure(new UnsupportedOperationException("Only 'local' storage is supported by this provider."));
        }
        if (apiKey == null || apiKey.isBlank()) {
            return Uni.createFrom().failure(new IllegalArgumentException("API Key cannot be null or blank."));
        }
        if (modelId == null || modelId.isBlank()) {
            return Uni.createFrom().failure(new IllegalArgumentException("Model ID cannot be null or blank."));
        }
        if (version == null || version.isBlank()) {
            return Uni.createFrom().failure(new IllegalArgumentException("Version cannot be null or blank."));
        }
        if (data == null || data.length == 0) {
            return Uni.createFrom().failure(new IllegalArgumentException("Model data cannot be null or empty."));
        }

        return Uni.createFrom().item(() -> {
            try {
                String key = generateStorageKey(apiKey, modelId, version);
                Path filePath = Paths.get(localBasePath, key);
                
                // Ensure parent directory exists
                Files.createDirectories(filePath.getParent());

                LOG.debugf("Writing model data to %s", filePath);
                Files.write(filePath, data);
                
                return "file://" + filePath.toAbsolutePath().toString();
            } catch (IOException e) {
                LOG.error("Failed to upload model to local storage", e);
                throw new RuntimeException("Failed to upload model to local storage", e);
            }
        }).runSubscriptionOn(executorService);
    }

    @Override
    public Uni<byte[]> downloadModel(String storageUri) {
        if (storageUri == null || !storageUri.startsWith("file://")) {
            return Uni.createFrom().failure(new IllegalArgumentException("Invalid local storage URI: " + storageUri));
        }

        return Uni.createFrom().item(() -> {
            try {
                Path filePath = Paths.get(storageUri.substring("file://".length()));
                if (!Files.exists(filePath) || !Files.isReadable(filePath)) {
                    throw new IOException("Model file not found or not readable: " + filePath);
                }
                return Files.readAllBytes(filePath);
            } catch (IOException e) {
                LOG.errorf("Failed to download model from %s", storageUri, e);
                throw new RuntimeException("Failed to download model from local storage", e);
            }
        }).runSubscriptionOn(executorService);
    }

    @Override
    public Uni<Void> deleteModel(String storageUri) {
        if (storageUri == null || !storageUri.startsWith("file://")) {
            return Uni.createFrom().failure(new IllegalArgumentException("Invalid local storage URI: " + storageUri));
        }

        return Uni.createFrom().item(() -> {
            try {
                Path filePath = Paths.get(storageUri.substring("file://".length()));
                if (Files.exists(filePath)) {
                    Files.delete(filePath);
                    LOG.infof("Deleted model file: %s", filePath);

                    // Cleanup empty parent directories up to base path
                    cleanupEmptyParents(filePath.getParent());
                } else {
                    LOG.warnf("Attempted to delete non-existent file: %s", filePath);
                }
                return null;
            } catch (IOException e) {
                LOG.errorf("Failed to delete model from %s", storageUri, e);
                throw new RuntimeException("Failed to delete model from local storage", e);
            }
        }).runSubscriptionOn(executorService).replaceWithVoid();
    }

    /**
     * Check if a model exists at the given URI.
     */
    public Uni<Boolean> modelExists(String storageUri) {
        if (storageUri == null || !storageUri.startsWith("file://")) {
            return Uni.createFrom().item(false);
        }
        return Uni.createFrom().item(() -> {
            Path filePath = Paths.get(storageUri.substring("file://".length()));
            return Files.exists(filePath);
        }).runSubscriptionOn(executorService);
    }

    private void cleanupEmptyParents(Path directory) throws IOException {
        Path base = Paths.get(localBasePath);
        Path current = directory;
        while (current != null && !current.equals(base) && current.startsWith(base)) {
            if (Files.isDirectory(current) && isDirEmpty(current)) {
                Files.delete(current);
                LOG.debugf("Deleted empty directory: %s", current);
                current = current.getParent();
            } else {
                break;
            }
        }
    }

    private boolean isDirEmpty(Path path) throws IOException {
        try (Stream<Path> entries = Files.list(path)) {
            return !entries.findFirst().isPresent();
        }
    }

    /**
     * Generate a consistent storage key/path within the local base path.
     */
    private String generateStorageKey(String apiKey, String modelId, String version) {
        // Includes UUID to ensure uniqueness even for same version uploads
        return String.format("%s/%s/%s/%s.bin", apiKey, modelId, version, UUID.randomUUID());
    }
}