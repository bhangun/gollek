package tech.kayys.golek.model.core;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.model.ModelManifest;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Disk-based implementation of ModelRepository
 */
@ApplicationScoped
public class LocalModelRepository implements ModelRepository {

    private static final Logger LOG = Logger.getLogger(LocalModelRepository.class);
    private static final String MANIFEST_FILE = "manifest.json";

    private final Path rootPath;
    private final ObjectMapper objectMapper;

    @Inject
    public LocalModelRepository(
            @ConfigProperty(name = "golek.model.repo.path", defaultValue = "models") String repoPath) {
        this.rootPath = Path.of(repoPath);
        this.objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());
        ensureDirectory(rootPath);
    }

    @Override
    public Uni<ModelManifest> findById(String modelId, String tenantId) {
        return Uni.createFrom().item(() -> {
            Path manifestPath = getManifestPath(modelId, tenantId);
            if (!Files.exists(manifestPath)) {
                return null;
            }
            try {
                return objectMapper.readValue(manifestPath.toFile(), ModelManifest.class);
            } catch (IOException e) {
                LOG.errorf(e, "Failed to read manifest for model %s", modelId);
                return null;
            }
        });
    }

    @Override
    public Uni<List<ModelManifest>> list(String tenantId, Pageable pageable) {
        return Uni.createFrom().item(() -> {
            Path tenantPath = rootPath.resolve(tenantId);
            if (!Files.exists(tenantPath)) {
                return List.of();
            }

            List<ModelManifest> manifests = new ArrayList<>();
            try (Stream<Path> modelDirs = Files.list(tenantPath)) {
                modelDirs.filter(Files::isDirectory)
                        .forEach(dir -> {
                            Path manifestPath = dir.resolve(MANIFEST_FILE);
                            if (Files.exists(manifestPath)) {
                                try {
                                    manifests.add(objectMapper.readValue(manifestPath.toFile(), ModelManifest.class));
                                } catch (IOException e) {
                                    LOG.warnf("Failed to read manifest in %s: %s", dir, e.getMessage());
                                }
                            }
                        });
            } catch (IOException e) {
                LOG.errorf(e, "Failed to list models for tenant %s", tenantId);
            }

            // Apply pagination
            int start = Math.min(pageable.offset(), manifests.size());
            int end = Math.min(start + pageable.size(), manifests.size());
            return manifests.subList(start, end);
        });
    }

    @Override
    public Uni<ModelManifest> save(ModelManifest manifest) {
        return Uni.createFrom().item(() -> {
            Path modelDir = rootPath.resolve(manifest.tenantId()).resolve(manifest.modelId());
            ensureDirectory(modelDir);

            Path manifestPath = modelDir.resolve(MANIFEST_FILE);
            Path tempManifestPath = modelDir.resolve(MANIFEST_FILE + ".tmp");
            Path lockPath = modelDir.resolve(".lock");

            try {
                // Try to acquire lock
                if (!acquireLock(lockPath)) {
                    throw new RuntimeException("Could not acquire lock for model: " + manifest.modelId());
                }

                try {
                    // Atomic write: Write to .tmp then move
                    objectMapper.writerWithDefaultPrettyPrinter().writeValue(tempManifestPath.toFile(), manifest);
                    Files.move(tempManifestPath, manifestPath, StandardCopyOption.REPLACE_EXISTING,
                            StandardCopyOption.ATOMIC_MOVE);
                    LOG.infof("Saved manifest for model %s (atomically)", manifest.modelId());
                    return manifest;
                } finally {
                    releaseLock(lockPath);
                }
            } catch (IOException e) {
                LOG.errorf(e, "Failed to save manifest for model %s", manifest.modelId());
                throw new RuntimeException("Failed to save model manifest", e);
            }
        });
    }

    private boolean acquireLock(Path lockPath) throws IOException {
        int retries = 5;
        while (retries > 0) {
            try {
                Files.createFile(lockPath);
                return true;
            } catch (IOException e) {
                retries--;
                try {
                    Thread.sleep(200);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IOException(ie);
                }
            }
        }
        return false;
    }

    private void releaseLock(Path lockPath) {
        try {
            Files.deleteIfExists(lockPath);
        } catch (IOException e) {
            LOG.warnf("Failed to release lock: %s", lockPath);
        }
    }

    @Override
    public Uni<Void> delete(String modelId, String tenantId) {
        return Uni.createFrom().item(() -> {
            Path modelDir = rootPath.resolve(tenantId).resolve(modelId);
            if (Files.exists(modelDir)) {
                try (Stream<Path> files = Files.walk(modelDir)) {
                    files.sorted((a, b) -> b.compareTo(a)) // Delete files before directories
                            .forEach(p -> {
                                try {
                                    Files.delete(p);
                                } catch (IOException e) {
                                    LOG.warnf("Failed to delete %s: %s", p, e.getMessage());
                                }
                            });
                } catch (IOException e) {
                    LOG.errorf(e, "Failed to delete model directory %s", modelDir);
                    throw new RuntimeException("Failed to delete model", e);
                }
            }
            return null;
        });
    }

    private Path getManifestPath(String modelId, String tenantId) {
        return rootPath.resolve(tenantId).resolve(modelId).resolve(MANIFEST_FILE);
    }

    private void ensureDirectory(Path path) {
        try {
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to create directory: " + path, e);
        }
    }
}
