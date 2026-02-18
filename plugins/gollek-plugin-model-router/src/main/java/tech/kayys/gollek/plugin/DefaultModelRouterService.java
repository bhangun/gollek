package tech.kayys.gollek.plugin;

import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Default implementation of ModelRouterService that uses scoring algorithms
 * to determine the optimal provider for a given model.
 */
@ApplicationScoped
public class DefaultModelRouterService implements ModelRouterService {

    // In-memory store for provider availability and capabilities
    private final Map<String, Set<String>> modelToProviders = new ConcurrentHashMap<>();
    private final Map<String, ProviderCapabilities> providerCapabilities = new ConcurrentHashMap<>();

    // Scoring weights for different factors
    private static final double PERFORMANCE_WEIGHT = 0.3;
    private static final double COST_WEIGHT = 0.25;
    private static final double LATENCY_WEIGHT = 0.25;
    private static final double RELIABILITY_WEIGHT = 0.2;

    public DefaultModelRouterService() {
        initializeDefaultProviders();
    }

    private void initializeDefaultProviders() {
        // Register some default providers for common models
        registerProviderForModel("gpt-4", "openai-provider");
        registerProviderForModel("gpt-3.5-turbo", "openai-provider");
        registerProviderForModel("claude-3", "anthropic-provider");
        registerProviderForModel("llama-2", "llama-provider");
        registerProviderForModel("mistral-7b", "mistral-provider");

        // Set capabilities for each provider
        providerCapabilities.put("openai-provider", new ProviderCapabilities(
                "openai-provider", 99.9, 120, 0.05, "high"));
        providerCapabilities.put("anthropic-provider", new ProviderCapabilities(
                "anthropic-provider", 99.8, 150, 0.07, "medium"));
        providerCapabilities.put("llama-provider", new ProviderCapabilities(
                "llama-provider", 99.5, 80, 0.02, "low"));
        providerCapabilities.put("mistral-provider", new ProviderCapabilities(
                "mistral-provider", 99.7, 90, 0.03, "low"));
    }

    private void registerProviderForModel(String modelId, String providerId) {
        modelToProviders.computeIfAbsent(modelId, k -> ConcurrentHashMap.newKeySet()).add(providerId);
    }

    @Override
    public String selectProvider(String modelId, String requestId) {
        return selectProvider(modelId, requestId, Map.of());
    }

    @Override
    public String selectProvider(String modelId, String requestId, Map<String, Object> requestContext) {
        List<String> availableProviders = getAvailableProviders(modelId);

        if (availableProviders.isEmpty()) {
            // Try to find a fallback provider if no direct match exists
            return findFallbackProvider(modelId);
        }

        // Score each provider and select the best one
        return availableProviders.stream()
                .map(providerId -> {
                    double score = calculateScore(modelId, providerId, requestId, requestContext);
                    return new ProviderScore(providerId, score);
                })
                .max((a, b) -> Double.compare(a.score, b.score))
                .map(providerScore -> providerScore.providerId)
                .orElse(findFallbackProvider(modelId)); // Use fallback if scoring fails
    }

    private String findFallbackProvider(String modelId) {
        // Implement fallback logic
        // For example, if we have a generic provider that can handle most models
        if (modelId.toLowerCase().contains("gpt")) {
            return "openai-provider";
        } else if (modelId.toLowerCase().contains("claude")) {
            return "anthropic-provider";
        } else if (modelId.toLowerCase().contains("llama")) {
            return "llama-provider";
        } else if (modelId.toLowerCase().contains("mistral")) {
            return "mistral-provider";
        }

        // Return the first available provider as ultimate fallback
        List<String> allProviders = List.of("openai-provider", "anthropic-provider", "llama-provider",
                "mistral-provider");
        return allProviders.stream()
                .filter(providerCapabilities::containsKey)
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<String> getAvailableProviders(String modelId) {
        Set<String> providers = modelToProviders.get(modelId);
        return providers != null ? List.copyOf(providers) : List.of();
    }

    @Override
    public double getProviderScore(String modelId, String providerId, String requestId) {
        return calculateScore(modelId, providerId, requestId, Map.of());
    }

    @Override
    public RoutingDecision getRoutingDecision(String modelId, String requestId, Map<String, Object> context) {
        String selectedProvider = selectProvider(modelId, requestId, context);
        List<String> candidates = getAvailableProviders(modelId);
        double score = selectedProvider != null ? getProviderScore(modelId, selectedProvider, requestId) : 0.0;

        return RoutingDecision.builder()
                .modelId(modelId)
                .providerId(selectedProvider)
                .requestId(requestId)
                .score(score)
                .candidates(candidates)
                .build();
    }

    private double calculateScore(String modelId, String providerId, String requestId,
            Map<String, Object> requestContext) {
        ProviderCapabilities caps = providerCapabilities.get(providerId);
        if (caps == null) {
            return 0.0; // Invalid provider
        }

        // Get individual factor scores (normalized to 0-1 range)
        double performanceScore = normalizePerformance(caps.performance());
        double costScore = 1.0 - normalizeCost(caps.costPerThousand()); // Lower cost is better
        double latencyScore = 1.0 - normalizeLatency(caps.avgLatencyMs()); // Lower latency is better
        double reliabilityScore = normalizeReliability(caps.reliability());

        // Apply weights and calculate composite score
        double weightedScore = performanceScore * PERFORMANCE_WEIGHT +
                costScore * COST_WEIGHT +
                latencyScore * LATENCY_WEIGHT +
                reliabilityScore * RELIABILITY_WEIGHT;

        // Adjust score based on tenant preferences if available
        if (requestContext.containsKey("priority")) {
            String priority = (String) requestContext.get("priority");
            if ("performance".equals(priority)) {
                weightedScore *= 1.1; // Boost performance-focused providers
            } else if ("cost".equals(priority)) {
                weightedScore *= 1.05; // Slight boost for cost-focused providers
            }
        }

        // Ensure score stays within bounds
        return Math.max(0.0, Math.min(1.0, weightedScore));
    }

    private double normalizePerformance(double performance) {
        // Normalize performance metric (assuming it's out of 100)
        return Math.min(1.0, performance / 100.0);
    }

    private double normalizeCost(double costPerThousand) {
        // Normalize cost (assuming typical range is $0.01-$0.10 per thousand tokens)
        return Math.min(1.0, costPerThousand / 0.10);
    }

    private double normalizeLatency(double latencyMs) {
        // Normalize latency (assuming typical range is 50-500ms)
        return Math.min(1.0, latencyMs / 500.0);
    }

    private double normalizeReliability(double reliability) {
        // Normalize reliability (assuming it's out of 100)
        return Math.min(1.0, reliability / 100.0);
    }

    private record ProviderScore(String providerId, double score) {
    }

    public void registerModelProvider(String modelId, String providerId) {
        registerProviderForModel(modelId, providerId);
    }
}