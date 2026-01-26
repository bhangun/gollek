package tech.kayys.golek.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.annotation.PostConstruct;

import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Default implementation of ContentModerator that uses keyword-based detection
 * and basic heuristics to identify potentially unsafe content.
 */
@ApplicationScoped
public class DefaultContentModerator implements ContentModerator {

    // Configuration
    private ContentSafetyConfig config;

    // Common patterns for unsafe content
    private static final Set<String> ALL_CATEGORIES = Set.of(
        "hate_speech", "harassment", "violence", "self_harm",
        "sexual_content", "dangerous_content", "misinformation"
    );

    // Compile regex patterns for performance
    private static final Pattern HATE_SPEECH_PATTERN =
        Pattern.compile("\\b(hate|bigotry|rascist|nazi|fascist|slur)\\b", Pattern.CASE_INSENSITIVE);
    private static final Pattern VIOLENCE_PATTERN =
        Pattern.compile("\\b(kill|murder|assault|attack|violence|weapon|bomb|explosive)\\b", Pattern.CASE_INSENSITIVE);
    private static final Pattern SEXUAL_CONTENT_PATTERN =
        Pattern.compile("\\b(sexual|nude|porn|explicit|adult|nsfw)\\b", Pattern.CASE_INSENSITIVE);
    private static final Pattern SELF_HARM_PATTERN =
        Pattern.compile("\\b(suicide|self-harm|kill myself|end my life)\\b", Pattern.CASE_INSENSITIVE);

    @PostConstruct
    void init() {
        // Initialize with default configuration
        this.config = ContentSafetyConfig.builder().build();
    }

    public void setConfig(ContentSafetyConfig config) {
        this.config = config;
    }

    @Override
    public ModerationResult moderate(String content) {
        if (!config.isEnabled() || content == null || content.trim().isEmpty()) {
            return ModerationResult.safe();
        }

        // Convert to lowercase for case-insensitive matching
        String lowerContent = content.toLowerCase();

        // Check for prohibited patterns based on enabled categories
        Set<String> detectedCategories = detectCategories(lowerContent);

        if (!detectedCategories.isEmpty()) {
            // Calculate confidence based on number of violations and length of content
            double confidence = Math.min(0.95, 0.7 + (detectedCategories.size() * 0.1));

            return ModerationResult.unsafe(
                "Content contains potentially unsafe elements",
                detectedCategories,
                confidence
            );
        }

        return ModerationResult.safe();
    }

    @Override
    public ModerationResult moderate(String content, Set<String> categories) {
        if (!config.isEnabled() || content == null || content.trim().isEmpty()) {
            return ModerationResult.safe();
        }

        // Only check against specified categories
        String lowerContent = content.toLowerCase();
        Set<String> detectedCategories = categories.stream()
            .filter(cat -> matchesCategory(lowerContent, cat))
            .collect(Collectors.toSet());

        if (!detectedCategories.isEmpty()) {
            double confidence = Math.min(0.95, 0.7 + (detectedCategories.size() * 0.1));
            return ModerationResult.unsafe(
                "Content violates specified categories: " + String.join(", ", detectedCategories),
                detectedCategories,
                confidence
            );
        }

        return ModerationResult.safe();
    }

    private Set<String> detectCategories(String content) {
        Set<String> violations = Set.of();

        // Check each enabled category
        for (String category : config.getEnabledCategories()) {
            if (matchesCategory(content, category)) {
                java.util.HashSet<String> newSet = new java.util.HashSet<>(violations);
                newSet.add(category);
                violations = newSet;
            }
        }

        return violations;
    }

    private boolean matchesCategory(String content, String category) {
        switch (category.toLowerCase()) {
            case "hate_speech":
                return HATE_SPEECH_PATTERN.matcher(content).find();
            case "violence":
                return VIOLENCE_PATTERN.matcher(content).find();
            case "sexual_content":
                return SEXUAL_CONTENT_PATTERN.matcher(content).find();
            case "self_harm":
                return SELF_HARM_PATTERN.matcher(content).find();
            default:
                return false;
        }
    }
}