package tech.kayys.gollek.plugin.security;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.exception.PluginException;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.*;

/**
 * Plugin that detects and redacts PII from inference responses.
 * 
 * @since 1.0.0
 */
@ApplicationScoped
public class PIIRedactionResponsePlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(PIIRedactionResponsePlugin.class);
    
    private static final String PLUGIN_ID = "pii-redaction-response";
    private static final String VERSION = "1.0.0";

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String version() {
        return VERSION;
    }

    @Inject
    PIIRedactionService redactionService;

    private boolean enabled = true;
    private boolean redactResponses = true;
    private boolean auditEnabled = true;
    private Map<String, Object> config = new HashMap<>();

    @Override
    public InferencePhase phase() {
        return InferencePhase.POST_PROCESSING;
    }

    @Override
    public int order() {
        // Execute early in POST_PROCESSING to redact before returning to user
        return 10;
    }

    @Override
    public void initialize(tech.kayys.gollek.spi.plugin.PluginContext context) {
        this.enabled = Boolean.parseBoolean(context.getConfig("enabled", "true"));
        this.redactResponses = Boolean.parseBoolean(context.getConfig("redact-responses", "true"));
        this.auditEnabled = Boolean.parseBoolean(context.getConfig("audit-enabled", "true"));
        
        LOG.infof("Initialized %s plugin (enabled: %b, responses: %b, audit: %b)", 
            PLUGIN_ID, enabled, redactResponses, auditEnabled);
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        if (!enabled || !redactResponses) {
            return;
        }

        try {
            Optional<InferenceResponse> responseOpt = context.getVariable("response", InferenceResponse.class);
            if (responseOpt.isPresent()) {
                redactResponse(responseOpt.get(), context);
            }
        } catch (Exception e) {
            LOG.errorf(e, "Failed to execute PII redaction for response of request %s",
                context.token().getExecutionId());
        }
    }

    private void redactResponse(InferenceResponse response, ExecutionContext context) {
        if (response == null || response.getContent() == null) {
            return;
        }

        String originalContent = response.getContent();
        String redactedContent = redactionService.redact(originalContent);

        if (!originalContent.equals(redactedContent)) {
            if (auditEnabled) {
                Map<String, Integer> detections = redactionService.detectPII(originalContent);
                LOG.infof("PII detected in response for request %s: %s", 
                    response.getRequestId(), detections);
            }

            // Update response with redacted content
            InferenceResponse redactedResponse = response.toBuilder()
                .content(redactedContent)
                .metadata("pii_redacted", true)
                .build();

            context.putVariable("response", redactedResponse);
            LOG.infof("Redacted PII from response for request %s", response.getRequestId());
        }
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        this.redactResponses = (Boolean) newConfig.getOrDefault("redact-responses", true);
        this.auditEnabled = (Boolean) newConfig.getOrDefault("audit-enabled", true);

        LOG.infof("Updated %s configuration", PLUGIN_ID);
    }

    @Override
    public Map<String, Object> currentConfig() {
        Map<String, Object> current = new HashMap<>(config);
        current.put("enabled", enabled);
        current.put("redact-responses", redactResponses);
        current.put("audit-enabled", auditEnabled);
        return current;
    }
}
