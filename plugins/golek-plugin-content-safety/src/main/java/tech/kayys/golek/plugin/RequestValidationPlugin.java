package tech.kayys.golek.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.core.plugin.InferencePhasePlugin;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.PluginException;
import tech.kayys.golek.core.plugin.GolekConfigurablePlugin;

import org.jboss.logging.Logger;

import java.util.HashMap;
import java.util.Map;

/**
 * Plugin that validates inference requests.
 * Phase-bound to PRE_VALIDATE.
 */
@ApplicationScoped
public class RequestValidationPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(RequestValidationPlugin.class);
    private static final String PLUGIN_ID = "request-validation";

    private Map<String, Object> config = new HashMap<>();
    private int maxMessageLength = 10000;
    private int maxMessages = 100;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

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
    public void initialize(PluginContext context) {
        this.maxMessageLength = Integer.parseInt(context.getConfig("maxMessageLength", "10000"));
        this.maxMessages = Integer.parseInt(context.getConfig("maxMessages", "100"));

        LOG.infof("Initialized %s (maxMessageLength: %d, maxMessages: %d)",
                name(), maxMessageLength, maxMessages);
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        InferenceRequest request = (InferenceRequest) context.variables().get("request");
        if (request == null) {
            throw new IllegalStateException("Request not found in context");
        }

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
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) throws GolekConfigurablePlugin.ConfigurationException {
        this.config = new HashMap<>(newConfig);
        try {
            this.maxMessageLength = Integer.parseInt(newConfig.getOrDefault("maxMessageLength", 10000).toString());
            this.maxMessages = Integer.parseInt(newConfig.getOrDefault("maxMessages", 100).toString());
        } catch (NumberFormatException e) {
            throw new GolekConfigurablePlugin.ConfigurationException("Invalid configuration value", e);
        }
    }

    @Override
    public Map<String, Object> currentConfig() {
        return new HashMap<>(config);
    }

    public static class ValidationException extends RuntimeException {
        public ValidationException(String message) {
            super(message);
        }
    }
}