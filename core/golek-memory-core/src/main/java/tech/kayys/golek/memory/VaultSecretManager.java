package tech.kayys.golek.memory;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Vault integration for secret management
 * Production implementation should use actual Vault client
 */
@ApplicationScoped
public class VaultSecretManager {

    private static final Logger LOG = Logger.getLogger(VaultSecretManager.class);

    @ConfigProperty(name = "vault.enabled", defaultValue = "false")
    boolean vaultEnabled;

    @ConfigProperty(name = "vault.address")
    Optional<String> vaultAddress;

    @ConfigProperty(name = "vault.token")
    Optional<String> vaultToken;

    /**
     * Get secrets for a path
     */
    public Map<String, String> getSecrets(String path) {
        if (!vaultEnabled) {
            LOG.debug("Vault disabled, returning empty secrets");
            return Map.of();
        }

        LOG.debugf("Fetching secrets from Vault: %s", path);

        try {
            // TODO: Implement actual Vault integration
            // For now, return from environment variables
            return getSecretsFromEnvironment(path);

        } catch (Exception e) {
            LOG.errorf(e, "Failed to fetch secrets from Vault: %s", path);
            return Map.of();
        }
    }

    /**
     * Fallback to environment variables
     */
    private Map<String, String> getSecretsFromEnvironment(String path) {
        Map<String, String> secrets = new HashMap<>();
        String prefix = "VAULT_" + path.replace("/", "_").toUpperCase() + "_";

        System.getenv().forEach((key, value) -> {
            if (key.startsWith(prefix)) {
                String secretKey = key.substring(prefix.length()).toLowerCase();
                secrets.put(secretKey, value);
            }
        });

        return secrets;
    }
}