package tech.kayys.golek.spi.provider;

import java.util.Map;
import java.util.Set;

/**
 * Information about an available inference provider.
 * Used by SDK to expose provider details to users.
 */
public record ProviderInfo(
        String id,
        String name,
        String version,
        String description,
        String vendor,
        ProviderHealth.Status healthStatus,
        ProviderCapabilities capabilities,
        Set<String> supportedModels,
        Map<String, Object> metadata) {

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String id;
        private String name;
        private String version = "1.0.0";
        private String description;
        private String vendor;
        private ProviderHealth.Status healthStatus = ProviderHealth.Status.UNKNOWN;
        private ProviderCapabilities capabilities;
        private Set<String> supportedModels = Set.of();
        private Map<String, Object> metadata = Map.of();

        public Builder id(String id) {
            this.id = id;
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

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder vendor(String vendor) {
            this.vendor = vendor;
            return this;
        }

        public Builder healthStatus(ProviderHealth.Status healthStatus) {
            this.healthStatus = healthStatus;
            return this;
        }

        public Builder capabilities(ProviderCapabilities capabilities) {
            this.capabilities = capabilities;
            return this;
        }

        public Builder supportedModels(Set<String> supportedModels) {
            this.supportedModels = supportedModels;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public ProviderInfo build() {
            return new ProviderInfo(id, name, version, description, vendor,
                    healthStatus, capabilities, supportedModels, metadata);
        }
    }
}
