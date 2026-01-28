package tech.kayys.golek.core.inference;

import tech.kayys.golek.api.inference.InferencePhase;
import tech.kayys.golek.core.execution.ExecutionContext;

/**
 * Observer interface for inference lifecycle events.
 * Implementations provide metrics, tracing, and logging hooks.
 * 
 * All methods are invoked synchronously in the execution thread.
 * Implementations should be non-blocking and fast.
 */
public interface InferenceObserver {

    /**
     * Called when inference starts
     */
    void onStart(ExecutionContext context);

    /**
     * Called when a phase begins
     */
    void onPhase(InferencePhase phase, ExecutionContext context);

    /**
     * Called when inference completes successfully
     */
    void onSuccess(ExecutionContext context);

    /**
     * Called when inference fails
     */
    void onFailure(Throwable error, ExecutionContext context);

    /**
     * Called when a plugin executes (optional, for detailed tracing)
     */
    default void onPluginExecute(String pluginId, ExecutionContext context) {
        // Default: no-op
    }

    /**
     * Called when a provider is invoked
     */
    default void onProviderInvoke(String providerId, ExecutionContext context) {
        // Default: no-op
    }
}