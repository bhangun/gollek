package tech.kayys.gollek.plugin;

import java.util.Set;

/**
 * Configuration for content safety policies.
 */
public class ContentSafetyConfig {
    
    private final boolean enabled;
    private final Set<String> enabledCategories;
    private final double minimumConfidence;
    private final boolean blockOnAnyViolation;
    private final boolean logViolations;

    public ContentSafetyConfig(boolean enabled, Set<String> enabledCategories, 
                             double minimumConfidence, boolean blockOnAnyViolation, 
                             boolean logViolations) {
        this.enabled = enabled;
        this.enabledCategories = enabledCategories != null ? enabledCategories : Set.of();
        this.minimumConfidence = minimumConfidence;
        this.blockOnAnyViolation = blockOnAnyViolation;
        this.logViolations = logViolations;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public Set<String> getEnabledCategories() {
        return enabledCategories;
    }

    public double getMinimumConfidence() {
        return minimumConfidence;
    }

    public boolean isBlockOnAnyViolation() {
        return blockOnAnyViolation;
    }

    public boolean isLogViolations() {
        return logViolations;
    }

    public static ContentSafetyConfigBuilder builder() {
        return new ContentSafetyConfigBuilder();
    }

    public static class ContentSafetyConfigBuilder {
        private boolean enabled = true;
        private Set<String> enabledCategories = Set.of("hate_speech", "violence", "self_harm", "sexual_content");
        private double minimumConfidence = 0.7;
        private boolean blockOnAnyViolation = true;
        private boolean logViolations = true;

        public ContentSafetyConfigBuilder enabled(boolean enabled) {
            this.enabled = enabled;
            return this;
        }

        public ContentSafetyConfigBuilder enabledCategories(Set<String> enabledCategories) {
            this.enabledCategories = enabledCategories;
            return this;
        }

        public ContentSafetyConfigBuilder minimumConfidence(double minimumConfidence) {
            this.minimumConfidence = minimumConfidence;
            return this;
        }

        public ContentSafetyConfigBuilder blockOnAnyViolation(boolean blockOnAnyViolation) {
            this.blockOnAnyViolation = blockOnAnyViolation;
            return this;
        }

        public ContentSafetyConfigBuilder logViolations(boolean logViolations) {
            this.logViolations = logViolations;
            return this;
        }

        public ContentSafetyConfig build() {
            return new ContentSafetyConfig(enabled, enabledCategories, minimumConfidence, 
                                         blockOnAnyViolation, logViolations);
        }
    }
}