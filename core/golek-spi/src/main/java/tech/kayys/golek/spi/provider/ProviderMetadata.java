package tech.kayys.golek.spi.provider;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Objects;

/**
 * Provider metadata and identification
 */
public final class ProviderMetadata {

    private final String providerId;
    private final String name;
    private final String version;
    private final String description;
    private final String vendor;
    private final String homepage;

    @JsonCreator
    public ProviderMetadata(
            @JsonProperty("providerId") String providerId,
            @JsonProperty("name") String name,
            @JsonProperty("version") String version,
            @JsonProperty("description") String description,
            @JsonProperty("vendor") String vendor,
            @JsonProperty("homepage") String homepage) {
        this.providerId = Objects.requireNonNull(providerId, "providerId");
        this.name = Objects.requireNonNull(name, "name");
        this.version = Objects.requireNonNull(version, "version");
        this.description = description;
        this.vendor = vendor;
        this.homepage = homepage;
    }

    // Getters
    public String getProviderId() {
        return providerId;
    }

    public String getName() {
        return name;
    }

    public String getVersion() {
        return version;
    }

    public String getDescription() {
        return description;
    }

    public String getVendor() {
        return vendor;
    }

    public String getHomepage() {
        return homepage;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String providerId;
        private String name;
        private String version;
        private String description;
        private String vendor;
        private String homepage;

        public Builder providerId(String providerId) {
            this.providerId = providerId;
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

        public Builder homepage(String homepage) {
            this.homepage = homepage;
            return this;
        }

        public ProviderMetadata build() {
            return new ProviderMetadata(
                    providerId, name, version, description, vendor, homepage);
        }
    }

    @Override
    public String toString() {
        return "ProviderMetadata{" +
                "providerId='" + providerId + '\'' +
                ", name='" + name + '\'' +
                ", version='" + version + '\'' +
                '}';
    }
}