package tech.kayys.golek.api.model;

/**
 * Represents the location of a model artifact, including its URI and additional
 * metadata.
 */
public record ArtifactLocation(
        String uri,
        String checksum,
        Long size,
        String contentType) {

    public ArtifactLocation {
        if (uri == null || uri.isBlank()) {
            throw new IllegalArgumentException("URI cannot be null or blank");
        }
    }
}