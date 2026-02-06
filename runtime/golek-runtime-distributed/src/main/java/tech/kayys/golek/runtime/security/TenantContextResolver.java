package tech.kayys.golek.runtime.security;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.ws.rs.core.SecurityContext;
import tech.kayys.wayang.tenant.TenantContext;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.jwt.JsonWebToken;
import org.jboss.logging.Logger;

import java.util.Set;

/**
 * Resolves tenant context from JWT claims
 */
@ApplicationScoped
public class TenantContextResolver {

    private static final Logger LOG = Logger.getLogger(TenantContextResolver.class);

    @ConfigProperty(name = "wayang.multitenancy.enabled", defaultValue = "false")
    boolean multitenancyEnabled;

    public TenantContext resolve(SecurityContext securityContext) {
        if (!multitenancyEnabled) {
            String userId = securityContext != null && securityContext.getUserPrincipal() != null
                    ? securityContext.getUserPrincipal().getName()
                    : "anonymous";
            return TenantContext.builder()
                    .tenantId("default")
                    .userId(userId)
                    .build();
        }

        if (securityContext.getUserPrincipal() == null) {
            throw new UnauthorizedException("No user principal found");
        }

        if (securityContext.getUserPrincipal() instanceof JsonWebToken jwt) {
            String tenantId = jwt.getClaim("tenant_id");
            String userId = jwt.getSubject();
            Set<String> roles = jwt.getGroups();

            if (tenantId == null) {
                throw new UnauthorizedException("No tenant_id in JWT");
            }

            return TenantContext.builder()
                    .tenantId(tenantId)
                    .userId(userId)
                    .roles(roles)
                    .build();
        }

        throw new UnauthorizedException("Invalid security context");
    }
}
