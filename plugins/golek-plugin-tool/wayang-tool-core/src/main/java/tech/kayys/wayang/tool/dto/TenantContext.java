package tech.kayys.wayang.tool.dto;

import io.quarkus.security.identity.SecurityIdentity;
import jakarta.enterprise.context.RequestScoped;

import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * TenantContext - Request-scoped tenant information
 */
@RequestScoped
public class TenantContext {

    private static final Logger LOG = LoggerFactory.getLogger(TenantContext.class);

    private String tenantId;
    private String userId;
    private SecurityIdentity identity;

    public String getTenantId() {
        LOG.info("Tenant ID: {}", tenantId);
        if (tenantId == null) {
            throw new IllegalStateException("Tenant ID not set in context");
        }
        return tenantId;
    }

    public String getCurrentTenantId() {
        return getTenantId();
    }

    public void setTenantId(String tenantId) {
        this.tenantId = tenantId;
    }

    public String getUserId() {
        if (userId == null) {
            throw new IllegalStateException("User ID not set in context");
        }
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public SecurityIdentity getIdentity() {
        return identity;
    }

    public void setIdentity(SecurityIdentity identity) {
        this.identity = identity;
    }

    public boolean hasRole(String role) {
        return identity != null && identity.hasRole(role);
    }

    public Optional<String> getClaim(String claimName) {
        if (identity == null) {
            return Optional.empty();
        }
        return Optional.ofNullable(identity.getAttribute(claimName));
    }
}
