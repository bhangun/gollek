package tech.kayys.gollek.plugin;

import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;
import tech.kayys.gollek.spi.plugin.PhasePluginException;

import java.util.Map;

/**
 * Base interface for safety plugins.
 * Safety plugins validate content and enforce policies.
 */
public interface SafetyPlugin extends InferencePhasePlugin {

    /**
     * Validate content for safety issues.
     *
     * @param content Content to validate
     * @return Validation result
     */
    SafetyValidationResult validate(String content);

    @Override
    default int order() {
        return 0; // Default order
    }

    @Override
    default void onConfigUpdate(Map<String, Object> newConfig) throws PhasePluginException {
        // Default: no-op
    }

    @Override
    default Map<String, Object> currentConfig() {
        return Map.of();
    }

    /**
     * Safety validation result
     */
    record SafetyValidationResult(
            boolean isSafe,
            String reason,
            double confidence,
            java.util.List<SafetyViolation> violations) {

        public static SafetyValidationResult success() {
            return new SafetyValidationResult(
                    true,
                    "Content is safe",
                    1.0,
                    java.util.List.of());
        }

        public static SafetyValidationResult unsafe(
                String reason,
                java.util.List<SafetyViolation> violations) {
            return new SafetyValidationResult(
                    false,
                    reason,
                    0.9,
                    violations);
        }
    }

    /**
     * Safety violation record
     */
    record SafetyViolation(
            String category,
            String description,
            double severity,
            int position) {
    }
}