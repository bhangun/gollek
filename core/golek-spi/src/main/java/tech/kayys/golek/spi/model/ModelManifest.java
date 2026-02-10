package tech.kayys.golek.spi.model;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

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

    public ModelManifest {
        Objects.requireNonNull(modelId, "modelId cannot be null");
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(version, "version cannot be null");
        Objects.requireNonNull(tenantId, "tenantId cannot be null");
        artifacts = Optional.ofNullable(artifacts).orElse(Map.of());
        supportedDevices = Optional.ofNullable(supportedDevices).orElse(List.of());
        metadata = Optional.ofNullable(metadata).orElse(Map.of());
        createdAt = Optional.ofNullable(createdAt).orElse(Instant.now());
        updatedAt = Optional.ofNullable(updatedAt).orElse(Instant.now());
    }

    public static Builder builder() {
        return new Builder();
    }

    public boolean supportsFormat(ModelFormat format) {
        return artifacts != null && format != null && artifacts.containsKey(format);
    }

    public boolean supportsDevice(DeviceType deviceType) {
        return supportedDevices != null && deviceType != null && supportedDevices.stream()
                .anyMatch(d -> d.type() == deviceType);
    }

    public static class Builder {
        private String modelId;
        private String name;
        private String version;
        private String tenantId;
        private Map<ModelFormat, ArtifactLocation> artifacts;
        private List<SupportedDevice> supportedDevices;
        private ResourceRequirements resourceRequirements;
        private Map<String, Object> metadata;
        private Instant createdAt;
        private Instant updatedAt;

        Builder() {
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder tenantId(String tenantId) {
            this.tenantId = tenantId;
            return this;
        }

        public Builder artifacts(Map<ModelFormat, ArtifactLocation> artifacts) {
            this.artifacts = artifacts;
            return this;
        }

        public Builder supportedDevices(List<SupportedDevice> supportedDevices) {
            this.supportedDevices = supportedDevices;
            return this;
        }

        public Builder resourceRequirements(ResourceRequirements resourceRequirements) {
            this.resourceRequirements = resourceRequirements;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder createdAt(Instant createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder updatedAt(Instant updatedAt) {
            this.updatedAt = updatedAt;
            return this;
        }

        public ModelManifest build() {
            return new ModelManifest(modelId, name, version, tenantId, artifacts, supportedDevices, resourceRequirements,
                    metadata, createdAt, updatedAt);
        }

        @Override
        public String toString() {
            return "ModelManifest.Builder(modelId=" + this.modelId + ", name=" + this.name + ", version=" + this.version
                    + ", tenantId=" + this.tenantId + ", artifacts=" + this.artifacts + ", supportedDevices="
                    + this.supportedDevices + ", resourceRequirements=" + this.resourceRequirements + ", metadata="
                    + this.metadata + ", createdAt=" + this.createdAt + ", updatedAt=" + this.updatedAt + ")";
        }
    }
}
