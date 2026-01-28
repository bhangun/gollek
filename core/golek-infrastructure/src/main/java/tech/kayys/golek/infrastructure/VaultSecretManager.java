package tech.kayys.golek.infrastructure;

import io.quarkus.vault.VaultKVSecretEngine;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.api.tenant.TenantId;
import java.util.Map;

/**
 * Secure credential management via HashiCorp Vault.
 */
@ApplicationScoped
public class VaultSecretManager {

        @Inject
        VaultKVSecretEngine vault;

        public String getProviderApiKey(
                        TenantId tenantId,
                        String providerId) {
                String path = String.format(
                                "tenants/%s/providers/%s",
                                tenantId.value(),
                                providerId);

                return vault.readSecret(path)
                                .get("api_key");
        }

        public void storeProviderApiKey(
                        TenantId tenantId,
                        String providerId,
                        String apiKey) {
                String path = String.format(
                                "tenants/%s/providers/%s",
                                tenantId.value(),
                                providerId);

                vault.writeSecret(path, Map.of("api_key", apiKey));
        }
}