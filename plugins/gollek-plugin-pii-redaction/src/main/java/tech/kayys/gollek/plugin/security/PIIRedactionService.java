package tech.kayys.gollek.plugin.security;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Service for detecting and redacting Personally Identifiable Information (PII).
 * 
 * <p>Supports detection of:</p>
 * <ul>
 *   <li>Email addresses</li>
 *   <li>Phone numbers (international format)</li>
 *   <li>Credit card numbers</li>
 *   <li>Social Security Numbers (SSN)</li>
 *   <li>IP addresses</li>
 *   <li>Custom regex patterns</li>
 * </ul>
 * 
 * <h2>Configuration</h2>
 * <pre>
 * gollek:
 *   plugins:
 *     pii-redaction:
 *       enabled: true
 *       patterns:
 *         email:
 *           enabled: true
 *           regex: "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
 *           replacement: "[REDACTED_EMAIL]"
 *         phone:
 *           enabled: true
 *           regex: "\\+?[0-9]{10,15}"
 *           replacement: "[REDACTED_PHONE]"
 *         credit-card:
 *           enabled: true
 *           regex: "\\b[0-9]{13,19}\\b"
 *           replacement: "[REDACTED_CC]"
 *         ssn:
 *           enabled: true
 *           regex: "\\b[0-9]{3}-[0-9]{2}-[0-9]{4}\\b"
 *           replacement: "[REDACTED_SSN]"
 * </pre>
 * 
 * @since 1.0.0
 */
@ApplicationScoped
public class PIIRedactionService {

    private static final Logger LOG = Logger.getLogger(PIIRedactionService.class);

    private final Map<String, PIIPattern> patterns = new ConcurrentHashMap<>();
    private final Map<String, Integer> redactionStats = new ConcurrentHashMap<>();

    /**
     * Initialize with default patterns.
     */
    public PIIRedactionService() {
        initializeDefaultPatterns();
    }

    private void initializeDefaultPatterns() {
        // Email
        addPattern("email", 
            Pattern.compile("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"),
            "[REDACTED_EMAIL]",
            true);

        // Phone (international format)
        addPattern("phone",
            Pattern.compile("\\+?[\\d\\s\\-\\(\\)]{10,}"),
            "[REDACTED_PHONE]",
            true);

        // Credit Card (13-19 digits, with optional spaces/dashes)
        addPattern("credit-card",
            Pattern.compile("\\b[\\d\\s\\-]{13,19}\\b"),
            "[REDACTED_CC]",
            true);

        // SSN (US Social Security Number)
        addPattern("ssn",
            Pattern.compile("\\b[0-9]{3}-[0-9]{2}-[0-9]{4}\\b"),
            "[REDACTED_SSN]",
            true);

        // IP Address (IPv4)
        addPattern("ip-address",
            Pattern.compile("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b"),
            "[REDACTED_IP]",
            true);

        // API Keys / Secrets (common patterns)
        addPattern("api-key",
            Pattern.compile("\\b(?:api[_-]?key|apikey|secret[_-]?key)\\s*[=:]\\s*['\"]?[\\w\\-]{20,}['\"]?", Pattern.CASE_INSENSITIVE),
            "[REDACTED_API_KEY]",
            true);

        // AWS Access Key
        addPattern("aws-access-key",
            Pattern.compile("\\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\\b"),
            "[REDACTED_AWS_KEY]",
            true);

        LOG.info("Initialized PII redaction service with default patterns");
    }

    /**
     * Add a custom PII pattern.
     * 
     * @param name pattern name (e.g., "email", "phone")
     * @param regex regular expression pattern
     * @param replacement replacement string
     * @param enabled whether pattern is enabled
     */
    public void addPattern(String name, String regex, String replacement, boolean enabled) {
        addPattern(name, Pattern.compile(regex), replacement, enabled);
    }

    /**
     * Add a custom PII pattern.
     * 
     * @param name pattern name
     * @param pattern compiled regex pattern
     * @param replacement replacement string
     * @param enabled whether pattern is enabled
     */
    public void addPattern(String name, Pattern pattern, String replacement, boolean enabled) {
        patterns.put(name, new PIIPattern(pattern, replacement, enabled));
        LOG.debugf("Added PII pattern: %s (enabled: %b)", name, enabled);
    }

    /**
     * Enable or disable a pattern.
     * 
     * @param name pattern name
     * @param enabled enable/disable
     */
    public void setPatternEnabled(String name, boolean enabled) {
        PIIPattern existing = patterns.get(name);
        if (existing != null) {
            patterns.put(name, new PIIPattern(existing.pattern, existing.replacement, enabled));
            LOG.debugf("Pattern %s %s", name, enabled ? "enabled" : "disabled");
        }
    }

    /**
     * Redact PII from text.
     * 
     * @param text input text
     * @return text with PII redacted
     */
    public String redact(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }

        String result = text;
        int totalRedactions = 0;

        for (Map.Entry<String, PIIPattern> entry : patterns.entrySet()) {
            PIIPattern piiPattern = entry.getValue();
            
            if (!piiPattern.enabled) {
                continue;
            }

            Matcher matcher = piiPattern.pattern.matcher(result);
            int count = 0;
            
            if (matcher.find()) {
                result = matcher.replaceAll(piiPattern.replacement);
                count++;
            }

            if (count > 0) {
                totalRedactions += count;
                redactionStats.merge(entry.getKey(), count, Integer::sum);
            }
        }

        if (totalRedactions > 0) {
            LOG.debugf("Redacted %d PII occurrences from %d characters", totalRedactions, text.length());
        }

        return result;
    }

    /**
     * Detect PII in text without redacting.
     * 
     * @param text input text
     * @return map of pattern names to match counts
     */
    public Map<String, Integer> detectPII(String text) {
        Map<String, Integer> detections = new HashMap<>();

        if (text == null || text.isEmpty()) {
            return detections;
        }

        for (Map.Entry<String, PIIPattern> entry : patterns.entrySet()) {
            PIIPattern piiPattern = entry.getValue();
            
            if (!piiPattern.enabled) {
                continue;
            }

            Matcher matcher = piiPattern.pattern.matcher(text);
            int count = 0;
            
            while (matcher.find()) {
                count++;
            }

            if (count > 0) {
                detections.put(entry.getKey(), count);
            }
        }

        if (!detections.isEmpty()) {
            LOG.debugf("Detected PII in text: %s", detections);
        }

        return detections;
    }

    /**
     * Get redaction statistics.
     * 
     * @return map of pattern names to total redaction counts
     */
    public Map<String, Integer> getRedactionStats() {
        return Collections.unmodifiableMap(new HashMap<>(redactionStats));
    }

    /**
     * Clear redaction statistics.
     */
    public void clearStats() {
        redactionStats.clear();
        LOG.info("Cleared PII redaction statistics");
    }

    /**
     * Check if any PII patterns are enabled.
     * 
     * @return true if at least one pattern is enabled
     */
    public boolean hasEnabledPatterns() {
        return patterns.values().stream().anyMatch(p -> p.enabled);
    }

    /**
     * Get all configured patterns.
     * 
     * @return map of pattern names to pattern info
     */
    public Map<String, PatternInfo> getPatterns() {
        Map<String, PatternInfo> info = new HashMap<>();
        patterns.forEach((name, pattern) -> 
            info.put(name, new PatternInfo(name, pattern.replacement, pattern.enabled)));
        return info;
    }

    /**
     * PII pattern holder.
     */
    private static class PIIPattern {
        final Pattern pattern;
        final String replacement;
        final boolean enabled;

        PIIPattern(Pattern pattern, String replacement, boolean enabled) {
            this.pattern = pattern;
            this.replacement = replacement;
            this.enabled = enabled;
        }
    }

    /**
     * Pattern information DTO.
     */
    public static class PatternInfo {
        public final String name;
        public final String replacement;
        public final boolean enabled;

        public PatternInfo(String name, String replacement, boolean enabled) {
            this.name = name;
            this.replacement = replacement;
            this.enabled = enabled;
        }
    }
}
