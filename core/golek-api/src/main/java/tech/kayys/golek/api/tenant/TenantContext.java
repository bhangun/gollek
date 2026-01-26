package tech.kayys.golek.api.tenant;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.util.*;

/**
 * Immutable tenant context for multi-tenancy.
 */
public final class TenantContext {

    @NotBlank
    private final String tenantId;

    private final String userId;
    private final Set<String> roles;
    private final Map<String, String> attributes;

    @JsonCreator
    public TenantContext(
            @JsonProperty("tenantId") String tenantId,
            @JsonProperty("userId") String userId,
            @JsonProperty("roles") Set<String> roles,
            @JsonProperty("attributes") Map<String, String> attributes) {
        this.tenantId = Objects.requireNonNull(tenantId, "tenantId");
        this.userId = userId;
        this.roles = roles != null
                ? Collections.unmodifiableSet(new HashSet<>(roles))
                : Collections.emptySet();
        this.attributes = attributes != null
                ? Collections.unmodifiableMap(new HashMap<>(attributes))
                : Collections.emptyMap();
    }

    public String getTenantId() {
        return tenantId;
    }

    public Optional<String> getUserId() {
        return Optional.ofNullable(userId);
    }

    public Set<String> getRoles() {
        return roles;
    }

    public boolean hasRole(String role) {
        return roles.contains(role);
    }

    public Map<String, String> getAttributes() {
        return attributes;
    }

    public Optional<String> getAttribute(String key) {
        return Optional.ofNullable(attributes.get(key));
    }

    // Factory method
    public static TenantContext of(String tenantId) {
        return new TenantContext(tenantId, null, null, null);
    }

    public static TenantContext of(String tenantId, String userId) {
        return new TenantContext(tenantId, userId, null, null);
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String tenantId;
        private String userId;
        private final Set<String> roles = new HashSet<>();
        private final Map<String, String> attributes = new HashMap<>();

        public Builder tenantId(String tenantId) {
            this.tenantId = tenantId;
            return this;
        }

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Builder role(String role) {
            this.roles.add(role);
            return this;
        }

        public Builder roles(Set<String> roles) {
            this.roles.addAll(roles);
            return this;
        }

        public Builder attribute(String key, String value) {
            this.attributes.put(key, value);
            return this;
        }

        public Builder attributes(Map<String, String> attributes) {
            this.attributes.putAll(attributes);
            return this;
        }

        public TenantContext build() {
            Objects.requireNonNull(tenantId, "tenantId is required");
            return new TenantContext(tenantId, userId, roles, attributes);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof TenantContext that))
            return false;
        return tenantId.equals(that.tenantId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(tenantId);
    }

    @Override
    public String toString() {
        return "TenantContext{tenantId='" + tenantId + "', userId='" + userId + "'}";
    }
}