package tech.kayys.wayang.tool.entity;

import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import java.time.Instant;
import jakarta.persistence.Column;
import jakarta.persistence.Embedded;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.validation.constraints.NotNull;
import tech.kayys.wayang.tool.dto.AuthType;

/**
 * Authentication profile for API access
 * Secrets are stored in Vault, only references kept here
 */
@Entity
@Table(name = "mcp_auth_profiles")
public class AuthProfile extends PanacheEntityBase {

    @Id
    @Column(name = "profile_id")
    private String profileId;

    @NotNull
    @Column(name = "tenant_id")
    private String requestId;

    @NotNull
    @Column(name = "profile_name")
    private String profileName;

    @Enumerated(EnumType.STRING)
    @Column(name = "auth_type")
    private AuthType authType;

    @Column(name = "description")
    private String description;

    // Auth configuration (non-sensitive)
    @Embedded
    private AuthConfig config;

    // Vault reference for secrets
    @Column(name = "vault_path")
    private String vaultPath;

    @Column(name = "secret_key")
    private String secretKey;

    // Status
    @Column(name = "enabled")
    private boolean enabled = true;

    @Column(name = "expires_at")
    private Instant expiresAt;

    // Timestamps
    @Column(name = "created_at")
    private Instant createdAt;

    @Column(name = "updated_at")
    private Instant updatedAt;

    // Getters and Setters
    public String getProfileId() {
        return profileId;
    }

    public void setProfileId(String profileId) {
        this.profileId = profileId;
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public String getProfileName() {
        return profileName;
    }

    public void setProfileName(String profileName) {
        this.profileName = profileName;
    }

    public AuthType getAuthType() {
        return authType;
    }

    public void setAuthType(AuthType authType) {
        this.authType = authType;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public AuthConfig getConfig() {
        return config;
    }

    public void setConfig(AuthConfig config) {
        this.config = config;
    }

    public String getVaultPath() {
        return vaultPath;
    }

    public void setVaultPath(String vaultPath) {
        this.vaultPath = vaultPath;
    }

    public String getSecretKey() {
        return secretKey;
    }

    public void setSecretKey(String secretKey) {
        this.secretKey = secretKey;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public Instant getExpiresAt() {
        return expiresAt;
    }

    public void setExpiresAt(Instant expiresAt) {
        this.expiresAt = expiresAt;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }
}