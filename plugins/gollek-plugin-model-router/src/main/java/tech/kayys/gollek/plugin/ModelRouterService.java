package tech.kayys.gollek.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;

import java.util.List;
import java.util.Map;

/**
 * Service for intelligent model-to-provider routing decisions.
 * Selects the optimal provider for a given model based on various factors.
 */
@Default
@ApplicationScoped
public interface ModelRouterService {

    /**
     * Select the best provider for a given model and tenant
     */
    String selectProvider(String modelId, String requestId);

    /**
     * Select the best provider for a given model, tenant, and request context
     */
    String selectProvider(String modelId, String requestId, Map<String, Object> requestContext);

    /**
     * Get available providers for a model
     */
    List<String> getAvailableProviders(String modelId);

    /**
     * Get routing score for a provider for a specific model
     */
    double getProviderScore(String modelId, String providerId, String requestId);

    /**
     * Get routing decision details
     */
    RoutingDecision getRoutingDecision(String modelId, String requestId, Map<String, Object> context);
}