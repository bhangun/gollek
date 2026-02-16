package tech.kayys.wayang.tool.security;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.entity.AuthProfile;
import tech.kayys.wayang.tool.dto.HttpRequestContext;
import tech.kayys.wayang.security.secrets.vault.VaultSecretManager;

import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Auth injector - Injects authentication into HTTP requests
 */
@ApplicationScoped
public class AuthInjector {

    private static final Logger LOG = LoggerFactory.getLogger(AuthInjector.class);

    @jakarta.inject.Inject
    VaultSecretManager vaultManager;

    public Uni<HttpRequestContext> injectAuth(
            HttpRequestContext request,
            String authProfileId) {

        LOG.info("Injecting auth for request: {}", request);

        if (authProfileId == null) {
            return Uni.createFrom().item(request);
        }

        return AuthProfile.<AuthProfile>findById(authProfileId)
                .flatMap(profile -> {
                    if (profile == null || !profile.isEnabled()) {
                        return Uni.createFrom().item(request);
                    }

                    // Retrieve secret from Vault
                    return vaultManager.retrieve(
                            tech.kayys.wayang.security.secrets.dto.RetrieveSecretRequest.latest(
                                    profile.getRequestId(),
                                    profile.getVaultPath()))
                            .map(secret -> {
                                String secretValue = secret.data().get("auth_secret");
                                Map<String, String> headers = new HashMap<>(request.headers());

                                // Inject based on auth type and location
                                switch (profile.getConfig().getLocation()) {
                                    case HEADER -> {
                                        String headerValue = buildAuthHeader(
                                                profile.getConfig().getScheme(),
                                                secretValue);
                                        headers.put(
                                                profile.getConfig().getParamName(),
                                                headerValue);
                                    }
                                    case QUERY -> {
                                        // Add to query params (less secure)
                                        Map<String, String> queryParams = new HashMap<>(request.queryParams());
                                        queryParams.put(
                                                profile.getConfig().getParamName(),
                                                secretValue);
                                        return new HttpRequestContext(
                                                request.method(),
                                                request.url(),
                                                queryParams,
                                                headers,
                                                request.body(),
                                                request.contentType());
                                    }
                                }

                                return new HttpRequestContext(
                                        request.method(),
                                        request.url(),
                                        request.queryParams(),
                                        headers,
                                        request.body(),
                                        request.contentType());
                            });
                });
    }

    private String buildAuthHeader(String scheme, String secret) {
        if (scheme != null && !scheme.isEmpty()) {
            return scheme + " " + secret;
        }
        return secret;
    }
}
