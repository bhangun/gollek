package tech.kayys.golek.provider.core.loader;

import io.smallrye.mutiny.Uni;

import java.nio.file.Path;

/**
 * Resolves and downloads model artifacts
 */
public interface ArtifactResolver {

    /**
     * Resolve artifact location and download if needed
     */
    Uni<Path> resolve(String artifactId);

    /**
     * Check if artifact is available locally
     */
    boolean isAvailableLocally(String artifactId);

    /**
     * Get local path for artifact
     */
    Path getLocalPath(String artifactId);

    /**
     * Clear cached artifact
     */
    Uni<Void> clearCache(String artifactId);
}