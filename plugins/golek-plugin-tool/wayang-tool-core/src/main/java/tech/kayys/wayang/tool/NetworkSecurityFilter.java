package tech.kayys.wayang.mcp.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.mcp.domain.ToolGuardrails;
import tech.kayys.wayang.mcp.dto.HttpRequestContext;

/**
 * Network security filter
 */
@ApplicationScoped
public class NetworkSecurityFilter {

    private static final Logger LOG = LoggerFactory.getLogger(NetworkSecurityFilter.class);

    public Uni<Void> validateRequest(
            HttpRequestContext request,
            ToolGuardrails guardrails) {

        return Uni.createFrom().item(() -> {
            // Validate against allowed domains
            if (guardrails.getAllowedDomains() != null &&
                    !guardrails.getAllowedDomains().isEmpty()) {

                String domain = extractDomain(request.url());
                boolean allowed = guardrails.getAllowedDomains().stream()
                        .anyMatch(allowedDomain -> domain.endsWith(allowedDomain));

                if (!allowed) {
                    throw new SecurityException(
                            "Domain not allowed: " + domain);
                }
            }

            return null;
        });
    }

    private String extractDomain(String url) {
        try {
            java.net.URI uri = new java.net.URI(url);
            return uri.getHost();
        } catch (Exception e) {
            return "";
        }
    }
}
