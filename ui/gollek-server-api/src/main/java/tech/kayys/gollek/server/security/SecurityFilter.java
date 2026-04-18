package tech.kayys.gollek.server.security;

import jakarta.annotation.Priority;
import jakarta.inject.Inject;
import jakarta.ws.rs.Priorities;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.container.ContainerRequestFilter;
import jakarta.ws.rs.ext.Provider;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

@Provider
@Priority(Priorities.AUTHENTICATION)
public class SecurityFilter implements ContainerRequestFilter {

    private static final Logger LOG = Logger.getLogger(SecurityFilter.class);

    @Inject
    @ConfigProperty(name = "gollek.server.allowed-api-keys", defaultValue = "community")
    String allowedApiKeys;

    @Inject
    @ConfigProperty(name = "gollek.server.admin-secret", defaultValue = "admin-secret")
    String adminSecret;

    @Inject
    ApiKeyStore apiKeyStore;

    @Override
    public void filter(ContainerRequestContext requestContext) throws IOException {
        String path = requestContext.getUriInfo().getPath();
        // Allow unauthenticated access to health and open endpoints
        if (path.equals("health") || path.startsWith("q/")) {
            return;
        }

        // Admin endpoints require admin-secret header
        if (path.startsWith("v1/admin")) {
            String admin = requestContext.getHeaderString("X-ADMIN-SECRET");
            if (admin == null || !admin.equals(adminSecret)) {
                requestContext.abortWith(javax.ws.rs.core.Response.status(javax.ws.rs.core.Response.Status.FORBIDDEN)
                        .entity(java.util.Map.of("error", "Admin secret required")).build());
            }
            return;
        }

        String header = requestContext.getHeaderString("X-API-Key");
        if (header == null || header.isBlank()) {
            LOG.debug("Missing API key for " + path);
            requestContext.abortWith(javax.ws.rs.core.Response.status(javax.ws.rs.core.Response.Status.UNAUTHORIZED)
                    .entity(java.util.Map.of("error", "Missing API key")).build());
            return;
        }

        Set<String> allowed = apiKeyStore.listKeys();
        if (!allowed.contains(header)) {
            LOG.debug("Invalid API key for " + path);
            requestContext.abortWith(javax.ws.rs.core.Response.status(javax.ws.rs.core.Response.Status.FORBIDDEN)
                    .entity(java.util.Map.of("error", "Invalid API key")).build());
        }
    }
}
