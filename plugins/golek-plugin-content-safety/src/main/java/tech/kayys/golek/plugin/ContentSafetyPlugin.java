package tech.kayys.golek.plugin;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.golek.provider.core.inference.InferencePhasePlugin;
import tech.kayys.wayang.inference.api.InferenceRequest;
import tech.kayys.wayang.inference.api.Message;
import tech.kayys.wayang.inference.execution.ExecutionContext;
import tech.kayys.wayang.inference.pipeline.InferencePhase;
import tech.kayys.wayang.inference.plugin.*;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Plugin that checks content safety.
 * Phase-bound to VALIDATE.
 */
@ApplicationScoped
public class ContentSafetyPlugin implements InferencePhasePlugin, ConfigurablePlugin {

    private static final Logger LOG = Logger.getLogger(ContentSafetyPlugin.class);
    private static final String PLUGIN_ID = "content-safety";

    private Map<String, Object> config = new HashMap<>();
    private boolean enabled = true;
    private Set<String> blockedKeywords = new HashSet<>();
    private List<Pattern> blockedPatterns = new ArrayList<>();

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String name() {
        return "Content Safety Filter";
    }

    @Override
    public String version() {
        return "1.0.0";
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
    public Uni<Void> initialize(PluginContext context) {
        this.config = new HashMap<>(context.config());
        this.enabled = context.getConfigOrDefault("enabled", true);

        // Load blocked keywords
        @SuppressWarnings("unchecked")
        List<String> keywords = (List<String>) config.getOrDefault("blockedKeywords", List.of());
        this.blockedKeywords = new HashSet<>(keywords);

        // Load blocked patterns
        @SuppressWarnings("unchecked")
        List<String> patterns = (List<String>) config.getOrDefault("blockedPatterns", List.of());
        this.blockedPatterns = patterns.stream()
                .map(Pattern::compile)
                .toList();

        LOG.infof("Initialized %s (enabled: %s, keywords: %d, patterns: %d)",
                name(), enabled, blockedKeywords.size(), blockedPatterns.size());
        return Uni.createFrom().voidItem();
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        return enabled;
    }

    @Override
    public Uni<Void> execute(ExecutionContext context) {
        InferenceRequest request = context.getVariable("request", InferenceRequest.class)
                .orElseThrow(() -> new IllegalStateException("Request not found"));

        for (Message message : request.getMessages()) {
            String content = message.getContent().toLowerCase();

            // Check blocked keywords
            for (String keyword : blockedKeywords) {
                if (content.contains(keyword.toLowerCase())) {
                    throw new UnsafeContentException(
                            "Content contains blocked keyword: " + keyword);
                }
            }

            // Check blocked patterns
            for (Pattern pattern : blockedPatterns) {
                if (pattern.matcher(content).find()) {
                    throw new UnsafeContentException(
                            "Content matches blocked pattern: " + pattern.pattern());
                }
            }
        }

        LOG.debugf("Safety check passed for %s", request.getRequestId());
        return Uni.createFrom().voidItem();
    }

    @Override
    public boolean onFailure(ExecutionContext context, Throwable error) {
        if (error instanceof UnsafeContentException) {
            context.setError(error);
            return false; // Halt pipeline
        }
        return true; // Continue for other errors
    }

    @Override
    public Uni<Void> onConfigUpdate(Map<String, Object> newConfig) {
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

        return Uni.createFrom().voidItem();
    }

    @Override
    public Map<String, Object> currentConfig() {
        return new HashMap<>(config);
    }

    @Override
    public PluginMetadata metadata() {
        return PluginMetadata.builder()
                .id(id())
                .name(name())
                .version(version())
                .description("Filters unsafe content from inference requests")
                .author("Kayys Tech")
                .tag("safety")
                .tag("content-moderation")
                .build();
    }

    public static class UnsafeContentException extends RuntimeException {
        public UnsafeContentException(String message) {
            super(message);
        }
    }
}