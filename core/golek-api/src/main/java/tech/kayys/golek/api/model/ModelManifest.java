package tech.kayys.golek.api.model;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Immutable model manifest representing all metadata and artifacts
 * for a specific model version.
 */
public record ModelManifest(
        String modelId,
        String name,
        String version,
        String tenantId,
        Map<ModelFormat, ArtifactLocation> artifacts,
        List<SupportedDevice> supportedDevices,
        ResourceRequirements resourceRequirements,
        Map<String, Object> metadata,
        Instant createdAt,
        Instant updatedAt) {
    public boolean supportsFormat(ModelFormat format) {
        return artifacts.containsKey(format);
    }

    public boolean supportsDevice(DeviceType deviceType) {
        return supportedDevices.stream()
                .anyMatch(d -> d.type() == deviceType);
    }
}
