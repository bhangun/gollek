package tech.kayys.golek.runtime;

import java.time.Instant;
import java.util.List;
import java.util.Map;

import tech.kayys.golek.api.model.DeviceType;
import tech.kayys.golek.api.model.ModelFormat;
import tech.kayys.golek.api.tenant.TenantId;

/**
 * Immutable model manifest representing all metadata and artifacts
 * for a specific model version.
 */
public record ModelManifest(
        String modelId,
        String name,
        String version,
        TenantId tenantId,
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