package tech.kayys.golek.provider.core.loader;

import jakarta.enterprise.context.ApplicationScoped;
import java.io.File;
import java.util.Optional;

/**
 * Stubbed RemoteArtifactResolver.
 * Original implementation required missing Vert.x Web Client dependencies.
 */
@ApplicationScoped
public class RemoteArtifactResolver {

    public Optional<File> resolve(String uri) {
        return Optional.empty();
    }
}