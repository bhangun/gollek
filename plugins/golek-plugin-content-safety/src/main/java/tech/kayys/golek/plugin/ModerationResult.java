package tech.kayys.golek.plugin;

import java.util.Set;

/**
 * Result of content moderation analysis.
 */
public class ModerationResult {
    
    private final boolean safe;
    private final String reason;
    private final Set<String> categories;
    private final double confidence;

    public ModerationResult(boolean safe, String reason, Set<String> categories, double confidence) {
        this.safe = safe;
        this.reason = reason;
        this.categories = categories != null ? categories : Set.of();
        this.confidence = confidence;
    }

    public boolean isSafe() {
        return safe;
    }

    public String getReason() {
        return reason;
    }

    public Set<String> getCategories() {
        return categories;
    }

    public double getConfidence() {
        return confidence;
    }

    public static ModerationResult safe() {
        return new ModerationResult(true, "Content is safe", Set.of(), 1.0);
    }

    public static ModerationResult unsafe(String reason, Set<String> categories) {
        return new ModerationResult(false, reason, categories, 0.9); // High confidence in violation
    }

    public static ModerationResult unsafe(String reason, Set<String> categories, double confidence) {
        return new ModerationResult(false, reason, categories, confidence);
    }
}