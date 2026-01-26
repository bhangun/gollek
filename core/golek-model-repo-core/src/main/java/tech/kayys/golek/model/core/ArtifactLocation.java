package tech.kayys.golek.model.core;

import java.util.Map;

/**
 * Location and access metadata for a model artifact
 */
public record ArtifactLocation(
        String uri,
        String checksum,
        String provider,
        Map<String, String> params) {
}
