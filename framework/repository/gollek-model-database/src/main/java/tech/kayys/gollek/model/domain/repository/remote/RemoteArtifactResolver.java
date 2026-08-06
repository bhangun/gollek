package tech.kayys.gollek.model.domain.repository.remote;

import jakarta.enterprise.context.ApplicationScoped;
import java.io.File;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Optional;
import org.jboss.logging.Logger;

/**
 * Native Java HttpClient implementation of RemoteArtifactResolver.
 * Avoids the need for Vert.x Web Client dependencies.
 */
@ApplicationScoped
public class RemoteArtifactResolver {

    private static final Logger LOG = Logger.getLogger(RemoteArtifactResolver.class);
    
    private final HttpClient httpClient;

    public RemoteArtifactResolver() {
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
    }

    public Optional<File> resolve(String uri) {
        if (uri == null || uri.isBlank()) {
            return Optional.empty();
        }

        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(uri))
                    .GET()
                    .build();

            HttpResponse<InputStream> response = httpClient.send(request, HttpResponse.BodyHandlers.ofInputStream());

            if (response.statusCode() >= 200 && response.statusCode() < 300) {
                Path tempFile = Files.createTempFile("artifact-", ".tmp");
                try (InputStream is = response.body()) {
                    Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
                }
                tempFile.toFile().deleteOnExit();
                return Optional.of(tempFile.toFile());
            } else {
                LOG.warnf("Failed to resolve artifact from %s. HTTP Status: %d", uri, response.statusCode());
                return Optional.empty();
            }
        } catch (Exception e) {
            LOG.errorf(e, "Error resolving remote artifact from %s", uri);
            return Optional.empty();
        }
    }
}