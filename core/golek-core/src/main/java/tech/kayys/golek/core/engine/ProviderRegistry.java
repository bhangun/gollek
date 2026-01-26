package tech.kayys.golek.core.engine;

import java.util.List;
import java.util.Optional;

/**
 * Registry for inference providers.
 * Placeholder interface - actual implementation in provider modules.
 */
public interface ProviderRegistry {
    
    /**
     * Get all registered providers
     */
    List<Object> getAllProviders();
    
    /**
     * Get provider by ID
     */
    Optional<Object> getProvider(String providerId);
    
    /**
     * Check if provider is registered
     */
    boolean isRegistered(String providerId);
}
