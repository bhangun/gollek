package tech.kayys.gollek.spi.model;

import io.smallrye.mutiny.Uni;
import java.nio.file.Path;

public interface ArtifactResolver {
    Uni<Path> resolve(String artifactId);
    boolean isAvailableLocally(String artifactId);
    Path getLocalPath(String artifactId);
    Uni<Void> clearCache(String artifactId);
}
