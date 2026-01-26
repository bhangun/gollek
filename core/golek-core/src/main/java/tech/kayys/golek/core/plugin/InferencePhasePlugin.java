package tech.kayys.golek.core.plugin;

import tech.kayys.golek.api.inference.InferencePhase;
import tech.kayys.golek.core.engine.EngineContext;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.plugin.api.PluginException;

/**
 * Plugin that executes during a specific inference phase.
 * This is the primary extension point for custom logic.
 */
public interface InferencePhasePlugin extends GolekConfigurablePlugin {

    /**
     * The phase this plugin is bound to
     */
    InferencePhase phase();

    /**
     * Execute plugin logic for the current inference request.
     * 
     * @param context Execution context with mutable state
     * @param engine  Global engine context (read-only)
     * @throws PluginException if plugin execution fails
     */
    void execute(ExecutionContext context, EngineContext engine)
            throws PluginException;

    /**
     * Check if plugin should execute for this context.
     * Allows conditional execution based on request metadata.
     * 
     * Default: always execute
     */
    default boolean shouldExecute(ExecutionContext context) {
        return true;
    }

    /**
     * Plugin execution exception implementation.
     * Extends the API PluginException.
     */
    class PhasePluginException extends PluginException {
        private final boolean retryable;

        public PhasePluginException(String message) {
            this(message, null, false);
        }

        public PhasePluginException(String message, Throwable cause) {
            this(message, cause, false);
        }

        public PhasePluginException(String message, Throwable cause, boolean retryable) {
            super(message, cause);
            this.retryable = retryable;
        }

        public boolean isRetryable() {
            return retryable;
        }
    }
}