package tech.kayys.gollek.core.reliability;

import io.smallrye.mutiny.Uni;

/**
 * Strategy for handling fallback operations
 */
public interface FallbackStrategy {
    /**
     * Execute fallback operation
     */
    Uni<Object> execute(Object originalRequest, Throwable failure);
    
    /**
     * Check if this strategy can handle the given failure
     */
    boolean canHandle(Throwable failure);
    
    /**
     * Get the priority of this strategy (lower numbers execute first)
     */
    default int priority() {
        return 100;
    }
}