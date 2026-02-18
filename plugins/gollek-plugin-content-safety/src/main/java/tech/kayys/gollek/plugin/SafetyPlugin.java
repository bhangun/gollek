package tech.kayys.gollek.plugin;

import tech.kayys.gollek.core.plugin.InferencePhasePlugin;

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