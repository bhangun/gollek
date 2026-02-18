package tech.kayys.gollek.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.gollek.core.plugin.InferencePhasePlugin;
import tech.kayys.gollek.core.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.plugin.PluginContext;
import tech.kayys.gollek.spi.plugin.PluginException;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Plugin that checks content safety.
 * Phase-bound to VALIDATE.
 */
@ApplicationScoped
public class ContentSafetyPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(ContentSafetyPlugin.class);
    private static final String PLUGIN_ID = "tech.kayys/content-safety";

    private Map<String, Object> config = new HashMap<>();
    private boolean enabled = true;
    private Set<String> blockedKeywords = new HashSet<>();
    private List<Pattern> blockedPatterns = new ArrayList<>();

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.VALIDATE;
    }

    @Override
    public int order() {
        return 20;
    }

    @Override
    public void initialize(PluginContext context) {
        this.enabled = Boolean.parseBoolean(context.getConfig("enabled", "true"));

        LOG.infof("Initialized %s (enabled: %s)", PLUGIN_ID, enabled);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        return enabled;
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        InferenceRequest request = (InferenceRequest) context.variables().get("request");
        if (request == null) {
            throw new IllegalStateException("Request not found");
        }

        for (Message message : request.getMessages()) {
            String content = message.getContent();
            if (content == null)
                continue;

            String lowerContent = content.toLowerCase();

            // Check blocked keywords
            for (String keyword : blockedKeywords) {
                if (lowerContent.contains(keyword.toLowerCase())) {
                    throw new InferencePhasePlugin.PhasePluginException(
                            "Content contains blocked keyword: " + keyword);
                }
            }

            // Check blocked patterns
            for (Pattern pattern : blockedPatterns) {
                if (pattern.matcher(lowerContent).find()) {
                    throw new InferencePhasePlugin.PhasePluginException(
                            "Content matches blocked pattern: " + pattern.pattern());
                }
            }
        }

        LOG.debugf("Safety check passed for %s", request.getRequestId());
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);

        @SuppressWarnings("unchecked")
        List<String> keywords = (List<String>) newConfig.getOrDefault("blockedKeywords", List.of());
        this.blockedKeywords = new HashSet<>(keywords);

        @SuppressWarnings("unchecked")
        List<String> patterns = (List<String>) newConfig.getOrDefault("blockedPatterns", List.of());
        this.blockedPatterns = patterns.stream()
                .map(Pattern::compile)
                .toList();
    }

    @Override
    public Map<String, Object> currentConfig() {
        return new HashMap<>(config);
    }

    public static class UnsafeContentException extends RuntimeException {
        public UnsafeContentException(String message) {
            super(message);
        }
    }
}