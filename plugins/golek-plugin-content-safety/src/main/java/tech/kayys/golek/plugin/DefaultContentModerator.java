package tech.kayys.golek.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.core.plugin.GolekConfigurablePlugin;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.PluginException;

import tech.kayys.golek.plugin.SafetyPlugin.SafetyValidationResult;
import tech.kayys.golek.plugin.SafetyPlugin.SafetyViolation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/**
 * Default implementation of ContentModerator that uses keyword-based detection
 * and basic heuristics to identify potentially unsafe content.
 */
@ApplicationScoped
public class DefaultContentModerator implements ContentModerator {

    private static final Logger LOG = Logger.getLogger(DefaultContentModerator.class);

    private final Map<String, Object> config = new ConcurrentHashMap<>();
    private volatile List<Pattern> blockedPatterns = new ArrayList<>();

    // Compile regex patterns for performance
    private static final Pattern HATE_SPEECH_PATTERN = Pattern.compile("\\b(hate|bigotry|rascist|nazi|fascist|slur)\\b",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern VIOLENCE_PATTERN = Pattern
            .compile("\\b(kill|murder|assault|attack|violence|weapon|bomb|explosive)\\b", Pattern.CASE_INSENSITIVE);
    private static final Pattern SEXUAL_CONTENT_PATTERN = Pattern
            .compile("\\b(sexual|nude|porn|explicit|adult|nsfw)\\b", Pattern.CASE_INSENSITIVE);
    private static final Pattern SELF_HARM_PATTERN = Pattern
            .compile("\\b(suicide|self-harm|kill myself|end my life)\\b", Pattern.CASE_INSENSITIVE);

    @Override
    public String id() {
        return "tech.kayys/default-content-moderator";
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.VALIDATE;
    }

    @Override
    public void initialize(PluginContext context) {
        LOG.info("Initializing default content moderator");
        loadDefaultPatterns();
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        Object inputObj = context.variables().get("request");
        if (inputObj == null) {
            return;
        }

        String content = inputObj.toString();
        SafetyValidationResult result = validate(content);

        if (!result.isSafe()) {
            throw new PluginException("Content safety violation: " + result.reason());
        }
    }

    @Override
    public SafetyValidationResult validate(String content) {
        if (content == null || content.trim().isEmpty()) {
            return SafetyValidationResult.success();
        }

        List<SafetyViolation> violations = new ArrayList<>();
        String lowerContent = content.toLowerCase();

        if (HATE_SPEECH_PATTERN.matcher(lowerContent).find()) {
            violations.add(new SafetyViolation("hate_speech", "Detected hate speech", 0.9, -1));
        }
        if (VIOLENCE_PATTERN.matcher(lowerContent).find()) {
            violations.add(new SafetyViolation("violence", "Detected violence", 0.9, -1));
        }
        if (SEXUAL_CONTENT_PATTERN.matcher(lowerContent).find()) {
            violations.add(new SafetyViolation("sexual_content", "Detected sexual content", 0.9, -1));
        }
        if (SELF_HARM_PATTERN.matcher(lowerContent).find()) {
            violations.add(new SafetyViolation("self_harm", "Detected self-harm", 0.9, -1));
        }

        if (!violations.isEmpty()) {
            return SafetyValidationResult.unsafe(
                    "Content contains potentially unsafe elements",
                    violations);
        }

        return SafetyValidationResult.success();
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) throws GolekConfigurablePlugin.ConfigurationException {
        this.config.putAll(newConfig);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.copyOf(config);
    }

    private void loadDefaultPatterns() {
        // Already initialized as static constants for simplicity in this implementation
    }

    @Override
    public void shutdown() {
        LOG.info("Shutting down default content moderator");
    }
}