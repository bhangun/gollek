package tech.kayys.wayang.tool.dto;

import io.quarkus.security.identity.SecurityIdentity;
import jakarta.enterprise.context.RequestScoped;

import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * RequestContext - Request-scoped tenant information
 */
@RequestScoped
public class RequestContext {

    private static final Logger LOG = LoggerFactory.getLogger(RequestContext.class);

    private String requestId;
    private String userId;
    private SecurityIdentity identity;

    public String getRequestId() {
        LOG.info("Tenant ID: {}", requestId);
        if (requestId == null) {
            throw new IllegalStateException("Tenant ID not set in context");
        }
        return requestId;
    }

    public String getCurrentRequestId() {
        return getRequestId();
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
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
