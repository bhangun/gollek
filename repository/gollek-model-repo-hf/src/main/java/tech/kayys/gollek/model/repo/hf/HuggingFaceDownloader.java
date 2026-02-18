package tech.kayys.gollek.model.repo.hf;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.model.ModelArtifact;
import tech.kayys.gollek.spi.model.ModelDescriptor;

/**
 * Helper to download artifacts from HuggingFace.
 */
public class HuggingFaceDownloader {
    private static final Logger LOG = Logger.getLogger(HuggingFaceDownloader.class);
    private final HuggingFaceClient client;

    public HuggingFaceDownloader(HuggingFaceClient client) {
        this.client = client;
    }

    public ModelArtifact download(ModelDescriptor descriptor, Path targetDir) {
        try {
            String repo = descriptor.metadata().get("repo");
            String filename = descriptor.metadata().get("filename");
            
            Files.createDirectories(targetDir);
            
            if (filename != null && !filename.isBlank()) {
                Path targetFile = targetDir.resolve(filename);
                if (!Files.exists(targetFile)) {
                    client.downloadFile(repo, filename, targetFile, null);
                }
                return new ModelArtifact(targetFile, "sha256:unknown", descriptor.metadata());
            } else {
                // Download essential repository files
                client.downloadRepository(repo, targetDir, null);
                return new ModelArtifact(targetDir, "sha256:unknown", descriptor.metadata());
            }
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("Failed to download model from HuggingFace: " + descriptor.id(), e);
        }
    }
}
