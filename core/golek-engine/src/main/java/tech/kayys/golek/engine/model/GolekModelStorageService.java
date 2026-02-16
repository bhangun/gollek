package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import tech.kayys.golek.spi.storage.ModelStorageService;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.stream.Stream;

@ApplicationScoped
public class GolekModelStorageService implements ModelStorageService {

    @ConfigProperty(name = "golek.storage.provider", defaultValue = "local")
    String storageProvider;

    @ConfigProperty(name = "golek.storage.local.base-path", defaultValue = "/tmp/golek-models")
    String localBasePath;

    @Inject
    ExecutorService executorService; // Assuming an ExecutorService is available for async file ops

    @Override
    public Uni<String> uploadModel(String requestId, String modelId, String version, byte[] data) {
        if (!"local".equalsIgnoreCase(storageProvider)) {
            return Uni.createFrom()
                    .failure(new UnsupportedOperationException("Only 'local' storage is supported by this provider."));
        }
        if (requestId == null || requestId.isBlank()) {
            return Uni.createFrom().failure(new IllegalArgumentException("Tenant ID cannot be null or blank."));
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
                Path modelDir = Paths.get(localBasePath, requestId, modelId, version);
                Files.createDirectories(modelDir); // Ensure directory exists

                // Generate a unique filename for the model data within its version directory
                String filename = modelId + "-" + version + ".bin"; // Or use original file name if available
                Path filePath = modelDir.resolve(filename);

                Files.write(filePath, data);
                return "file://" + filePath.toAbsolutePath().toString();
            } catch (IOException e) {
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
                    // Delete the file itself
                    Files.delete(filePath);

                    // Attempt to delete parent directories if they become empty
                    // (requestId/modelId/version)
                    Path parent = filePath.getParent();
                    while (parent != null && !parent.equals(Paths.get(localBasePath))) {
                        if (Files.isDirectory(parent) && isEmpty(parent)) {
                            Files.delete(parent);
                            parent = parent.getParent();
                        } else {
                            break; // Stop if directory is not empty or we reached base path
                        }
                    }

                } else {
                    // Log a warning or just ignore if file doesn't exist, depending on desired
                    // behavior
                    System.out.println("Warning: Attempted to delete non-existent file: " + filePath);
                }
                return null;
            } catch (IOException e) {
                throw new RuntimeException("Failed to delete model from local storage", e);
            }
        }).runSubscriptionOn(executorService).replaceWithVoid();
    }

    // This helper is for the test only, not part of the SPI
    public Uni<Boolean> modelExists(String storageUri) {
        if (storageUri == null || !storageUri.startsWith("file://")) {
            return Uni.createFrom().item(false); // Or throw an IllegalArgumentException, depending on desired behavior
        }
        return Uni.createFrom().item(() -> {
            Path filePath = Paths.get(storageUri.substring("file://".length()));
            return Files.exists(filePath);
        }).runSubscriptionOn(executorService);
    }

    private boolean isEmpty(Path path) throws IOException {
        if (Files.isDirectory(path)) {
            try (Stream<Path> entries = Files.list(path)) {
                return !entries.findFirst().isPresent();
            }
        }
        return false;
    }

    // Helper to generate a consistent storage key/path within the local base path
    // Not part of the SPI, but useful for this implementation
    private String generateStorageKey(String requestId, String modelId, String version) {
        // This is a simplified example. A real system might use hashing, unique IDs,
        // etc.
        return requestId + "/" + modelId + "/" + version + "/" + UUID.randomUUID() + ".bin";
    }
}