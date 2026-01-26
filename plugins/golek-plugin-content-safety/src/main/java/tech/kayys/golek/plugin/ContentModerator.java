package tech.kayys.golek.provider.core.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/**
 * Content moderation safety plugin.
 * Validates both input and output content.
 */
@ApplicationScoped
public class ContentModerator implements SafetyPlugin {

    private static final Logger LOG = Logger.getLogger(ContentModerator.class);

    // Configurable patterns
    private final Map<String, Object> config = new ConcurrentHashMap<>();

    // Pre-compiled patterns for performance
    private volatile List<Pattern> blockedPatterns = new ArrayList<>();

    @Override
    public String id() {
        return "tech.kayys/content-moderator";
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.VALIDATE;
    }

    @Override
    public void initialize(EngineContext context) {
        LOG.info("Initializing content moderator");

        // Load default patterns
        loadDefaultPatterns();
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine)
            throws PluginException {

        // Get input messages
        Object inputObj = context.variables().get("request");
        if (inputObj == null) {
            return;
        }

        // Validate content
        String content = extractContent(inputObj);
        SafetyValidationResult result = validate(content);

        if (!result.safe()) {
            LOG.warnf("Content validation failed: %s", result.reason());
            throw new PluginException(
                    "Content safety validation failed: " + result.reason());
        }
    }

    @Override
    public SafetyValidationResult validate(String content) {
        if (content == null || content.isBlank()) {
            return SafetyValidationResult.safe();
        }

        List<SafetyViolation> violations = new ArrayList<>();

        // Check against patterns
        for (Pattern pattern : blockedPatterns) {
            var matcher = pattern.matcher(content);
            if (matcher.find()) {
                violations.add(new SafetyViolation(
                        "blocked_pattern",
                        "Content matches blocked pattern: " + pattern.pattern(),
                        0.9,
                        matcher.start()));
            }
        }

        // Check length
        if (content.length() > 100000) {
            violations.add(new SafetyViolation(
                    "excessive_length",
                    "Content exceeds maximum length",
                    0.8,
                    0));
        }

        if (!violations.isEmpty()) {
            return SafetyValidationResult.unsafe(
                    violations.size() + " safety violations detected",
                    violations);
        }

        return SafetyValidationResult.safe();
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig)
            throws ConfigurationException {

        this.config.putAll(newConfig);

        // Reload patterns if config changed
        if (newConfig.containsKey("blocked.patterns")) {
            loadPatternsFromConfig(newConfig);
        }
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.copyOf(config);
    }

    private void loadDefaultPatterns() {
        blockedPatterns = List.of(
                Pattern.compile("(?i)password\\s*[:=]\\s*[^\\s]+"),
                Pattern.compile("(?i)api[_-]?key\\s*[:=]\\s*[^\\s]+"),
                Pattern.compile("(?i)secret\\s*[:=]\\s*[^\\s]+"));
    }

    @SuppressWarnings("unchecked")
    private void loadPatternsFromConfig(Map<String, Object> config) {
        Object patterns = config.get("blocked.patterns");
        if (patterns instanceof List) {
            List<String> patternStrings = (List<String>) patterns;
            blockedPatterns = patternStrings.stream()
                    .map(Pattern::compile)
                    .toList();
        }
    }

    private String extractContent(Object input) {
        // Simple extraction - in real implementation, handle InferenceRequest
        return input.toString();
    }

    @Override
    public void shutdown() {
        LOG.info("Shutting down content moderator");
        blockedPatterns.clear();
    }
}