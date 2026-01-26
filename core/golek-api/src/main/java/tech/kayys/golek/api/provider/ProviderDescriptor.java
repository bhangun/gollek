package tech.kayys.golek.api.provider;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Metadata descriptor for a provider.
 */
public final class ProviderDescriptor {

    private final String id;
    private final String displayName;
    private final String version;
    private final String description;
    private final ProviderCapabilities capabilities;
    private final Map<String, Object> metadata;

    private ProviderDescriptor(Builder builder) {
        this.id = builder.id;
        this.displayName = builder.displayName;
        this.version = builder.version;
        this.description = builder.description;
        this.capabilities = builder.capabilities;
        this.metadata = Collections.unmodifiableMap(new HashMap<>(builder.metadata));
    }

    public String getId() {
        return id;
    }

    public String getDisplayName() {
        return displayName;
    }

    public String getVersion() {
        return version;
    }

    public String getDescription() {
        return description;
    }

    public ProviderCapabilities getCapabilities() {
        return capabilities;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String id;
        private String displayName;
        private String version = "1.0.0";
        private String description;
        private ProviderCapabilities capabilities;
        private final Map<String, Object> metadata = new HashMap<>();

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder displayName(String displayName) {
            this.displayName = displayName;
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

        public Builder capabilities(ProviderCapabilities capabilities) {
            this.capabilities = capabilities;
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public ProviderDescriptor build() {
            Objects.requireNonNull(id, "id is required");
            Objects.requireNonNull(capabilities, "capabilities is required");
            return new ProviderDescriptor(this);
        }
    }

    @Override
    public String toString() {
        return "ProviderDescriptor{" +
                "id='" + id + '\'' +
                ", displayName='" + displayName + '\'' +
                ", version='" + version + '\'' +
                '}';
    }
}