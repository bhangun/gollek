package tech.kayys.golek.plugin;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.api.InferenceRequest;
import tech.kayys.golek.api.Message;
import tech.kayys.golek.provider.core.plugin.GolekPlugin.PluginMetadata;

import org.jboss.logging.Logger;

import java.util.HashMap;
import java.util.Map;

/**
 * Plugin that validates inference requests.
 * Phase-bound to PRE_VALIDATE.
 */
@ApplicationScoped
public class RequestValidationPlugin implements InferencePhasePlugin, ConfigurablePlugin {

    private static final Logger LOG = Logger.getLogger(RequestValidationPlugin.class);
    private static final String PLUGIN_ID = "request-validation";

    private Map<String, Object> config = new HashMap<>();
    private int maxMessageLength = 10000;
    private int maxMessages = 100;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String name() {
        return "Request Validator";
    }

    @Override
    public String version() {
        return "1.0.0";
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.PRE_VALIDATE;
    }

    @Override
    public int order() {
        return 10; // Execute early
    }

    @Override
    public Uni<Void> initialize(PluginContext context) {
        this.config = new HashMap<>(context.config());
        this.maxMessageLength = context.getConfigOrDefault("maxMessageLength", 10000);
        this.maxMessages = context.getConfigOrDefault("maxMessages", 100);

        LOG.infof("Initialized %s (maxMessageLength: %d, maxMessages: %d)",
                name(), maxMessageLength, maxMessages);
        return Uni.createFrom().voidItem();
    }

    @Override
    public Uni<Void> execute(ExecutionContext context) {
        InferenceRequest request = context.getVariable("request", InferenceRequest.class)
                .orElseThrow(() -> new IllegalStateException("Request not found in context"));

        // Validate message count
        if (request.getMessages().size() > maxMessages) {
            throw new ValidationException(
                    String.format("Too many messages: %d (max: %d)",
                            request.getMessages().size(), maxMessages));
        }

        // Validate message lengths
        for (Message message : request.getMessages()) {
            if (message.getContent().length() > maxMessageLength) {
                throw new ValidationException(
                        String.format("Message too long: %d characters (max: %d)",
                                message.getContent().length(), maxMessageLength));
            }

            // Validate message content is not empty
            if (message.getContent().trim().isEmpty()) {
                throw new ValidationException("Message content cannot be empty");
            }
        }

        // Validate model name
        if (request.getModel() == null || request.getModel().isBlank()) {
            throw new ValidationException("Model name is required");
        }

        LOG.debugf("Request validation passed for %s", request.getRequestId());
        return Uni.createFrom().voidItem();
    }

    @Override
    public boolean onFailure(ExecutionContext context, Throwable error) {
        // Validation failures should halt the pipeline
        context.setError(error);
        return false;
    }

    @Override
    public Uni<Void> onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.maxMessageLength = (Integer) newConfig.getOrDefault("maxMessageLength", 10000);
        this.maxMessages = (Integer) newConfig.getOrDefault("maxMessages", 100);
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
                .description("Validates inference requests before processing")
                .author("Kayys Tech")
                .tag("validation")
                .tag("safety")
                .property("maxMessageLength", String.valueOf(maxMessageLength))
                .property("maxMessages", String.valueOf(maxMessages))
                .build();
    }

    public static class ValidationException extends RuntimeException {
        public ValidationException(String message) {
            super(message);
        }
    }
}