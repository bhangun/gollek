package tech.kayys.gollek.core.reliability;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for fallback strategies
 */
public class FallbackStrategyRegistry {
    private final Map<String, FallbackStrategy> strategies = new ConcurrentHashMap<>();

    /**
     * Register a fallback strategy
     */
    public void registerStrategy(String operationType, FallbackStrategy strategy) {
        strategies.put(operationType, strategy);
    }

    /**
     * Get a fallback strategy
     */
    public FallbackStrategy getStrategy(String operationType) {
        return strategies.get(operationType);
    }

    /**
     * Remove a fallback strategy
     */
    public void removeStrategy(String operationType) {
        strategies.remove(operationType);
    }

    /**
     * Check if a strategy exists
     */
    public boolean hasStrategy(String operationType) {
        return strategies.containsKey(operationType);
    }

    /**
     * Get all registered strategies
     */
    public Map<String, FallbackStrategy> getAllStrategies() {
        return new ConcurrentHashMap<>(strategies);
    }
}