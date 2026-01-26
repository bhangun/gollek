package tech.kayys.golek.provider.core.loader;

import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Resolves artifacts from local filesystem
 */
public class LocalArtifactResolver implements ArtifactResolver {

    private static final Logger LOG = Logger.getLogger(LocalArtifactResolver.class);

    private final Path basePath;
    private final Map<String, Path> cache = new ConcurrentHashMap<>();

    public LocalArtifactResolver(String basePath) {
        this.basePath = Paths.get(basePath);
        ensureDirectoryExists(this.basePath);
    }

    @Override
    public Uni<Path> resolve(String artifactId) {
        return Uni.createFrom().item(() -> {
            Path cached = cache.get(artifactId);
            if (cached != null && Files.exists(cached)) {
                LOG.debugf("Artifact %s found in cache: %s", artifactId, cached);
                return cached;
            }

            Path artifactPath = basePath.resolve(artifactId);

            if (!Files.exists(artifactPath)) {
                throw new RuntimeException(
                        "Artifact not found: " + artifactId + " at " + artifactPath);
            }

            cache.put(artifactId, artifactPath);
            LOG.infof("Artifact %s resolved to: %s", artifactId, artifactPath);

            return artifactPath;
        });
    }

    @Override
    public boolean isAvailableLocally(String artifactId) {
        if (cache.containsKey(artifactId)) {
            return true;
        }

        Path artifactPath = basePath.resolve(artifactId);
        return Files.exists(artifactPath);
    }

    @Override
    public Path getLocalPath(String artifactId) {
        Path cached = cache.get(artifactId);
        if (cached != null) {
            return cached;
        }
        return basePath.resolve(artifactId);
    }

    @Override
    public Uni<Void> clearCache(String artifactId) {
        return Uni.createFrom().item(() -> {
            cache.remove(artifactId);
            LOG.debugf("Cleared cache for artifact: %s", artifactId);
            return null;
        });
    }

    private void ensureDirectoryExists(Path path) {
        try {
            if (!Files.exists(path)) {
                Files.createDirectories(path);
                LOG.infof("Created artifact directory: %s", path);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to create artifact directory: " + path, e);
        }
    }
}