package tech.kayys.golek.model.core;

import java.nio.file.Path;
import java.util.Map;

public record ModelArtifact(
                Path path,
                String checksum,
                Map<String, String> metadata) {
}