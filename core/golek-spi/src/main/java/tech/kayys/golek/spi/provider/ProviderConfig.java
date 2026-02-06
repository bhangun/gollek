package tech.kayys.golek.spi.provider;

import java.time.Duration;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable provider configuration
 */
public final class ProviderConfig {

    private final String providerId;
    private final Map<String, Object> properties;
    private final Map<String, String> secrets;
    private final boolean enabled;
    private final int priority;
    private final Duration timeout;
    private final Map<String, Object> metadata;

    private ProviderConfig(Builder builder) {
        this.providerId = builder.providerId;
        this.properties = Collections.unmodifiableMap(
                new HashMap<>(builder.properties));
        this.secrets = Collections.unmodifiableMap(
                new HashMap<>(builder.secrets));
        this.enabled = builder.enabled;
        this.priority = builder.priority;
        this.timeout = builder.timeout;
        this.metadata = Collections.unmodifiableMap(
                new HashMap<>(builder.metadata));
    }

    public String getProviderId() {
        return providerId;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public int getPriority() {
        return priority;
    }

    public Duration getTimeout() {
        return timeout;
    }

    // Property getters
    public String getString(String key) {
        return getString(key, null);
    }

    public String getString(String key, String defaultValue) {
        Object value = properties.get(key);
        return value != null ? value.toString() : defaultValue;
    }

    public int getInt(String key, int defaultValue) {
        Object value = properties.get(key);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        if (value instanceof String) {
            try {
                return Integer.parseInt((String) value);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    public long getLong(String key, long defaultValue) {
        Object value = properties.get(key);
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        if (value instanceof String) {
            try {
                return Long.parseLong((String) value);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    public double getDouble(String key, double defaultValue) {
        Object value = properties.get(key);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        if (value instanceof String) {
            try {
                return Double.parseDouble((String) value);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    public boolean getBoolean(String key, boolean defaultValue) {
        Object value = properties.get(key);
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof String) {
            return Boolean.parseBoolean((String) value);
        }
        return defaultValue;
    }

    @SuppressWarnings("unchecked")
    public <T> Optional<T> get(String key, Class<T> type) {
        Object value = properties.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    public Map<String, Object> getProperties() {
        return properties;
    }

    // Secret getters
    public Optional<String> getSecret(String key) {
        return Optional.ofNullable(secrets.get(key));
    }

    public String getRequiredSecret(String key) {
        return getSecret(key)
                .orElseThrow(() -> new IllegalStateException(
                        "Required secret not found: " + key));
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static Builder builder(String providerId) {
        return new Builder().providerId(providerId);
    }

    public static class Builder {
        private String providerId;
        private final Map<String, Object> properties = new HashMap<>();
        private final Map<String, String> secrets = new HashMap<>();
        private boolean enabled = true;
        private int priority = 50;
        private Duration timeout = Duration.ofSeconds(30);
        private final Map<String, Object> metadata = new HashMap<>();

        public Builder providerId(String providerId) {
            this.providerId = providerId;
            return this;
        }

        public Builder property(String key, Object value) {
            this.properties.put(key, value);
            return this;
        }

        public Builder properties(Map<String, Object> properties) {
            this.properties.putAll(properties);
            return this;
        }

        public Builder secret(String key, String value) {
            this.secrets.put(key, value);
            return this;
        }

        public Builder secrets(Map<String, String> secrets) {
            this.secrets.putAll(secrets);
            return this;
        }

        public Builder enabled(boolean enabled) {
            this.enabled = enabled;
            return this;
        }

        public Builder priority(int priority) {
            this.priority = priority;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public ProviderConfig build() {
            Objects.requireNonNull(providerId, "providerId is required");
            return new ProviderConfig(this);
        }
    }

    @Override
    public String toString() {
        return "ProviderConfig{" +
                "providerId='" + providerId + '\'' +
                ", enabled=" + enabled +
                ", priority=" + priority +
                ", timeout=" + timeout +
                ", properties=" + properties.size() +
                ", secrets=" + secrets.size() +
                '}';
    }
}