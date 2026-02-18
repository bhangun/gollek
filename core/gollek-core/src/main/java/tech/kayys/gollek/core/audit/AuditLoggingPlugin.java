package tech.kayys.gollek.core.audit;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.AuditPayload;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.core.plugin.InferencePhasePlugin;
import tech.kayys.gollek.core.execution.ExecutionContext;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.plugin.PluginContext;

import org.jboss.logging.Logger;

import java.util.HashMap;
import java.util.Map;

/**
 * Plugin that logs audit events for all inferences.
 * Phase-bound to AUDIT.
 */
@ApplicationScoped
public class AuditLoggingPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(AuditLoggingPlugin.class);
    private static final String PLUGIN_ID = "audit-logging";

    // @Inject
    // AuditService auditService; // This service seems to be missing or internal

    private Map<String, Object> config = new HashMap<>();
    private boolean enabled = true;
    private boolean logInputs = true;
    private boolean logOutputs = true;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    // GollekPlugin now provides order, version, etc. from interface defaults or we
    // override

    @Override
    public InferencePhase phase() {
        return InferencePhase.AUDIT;
    }

    @Override
    public void initialize(PluginContext context) {
        this.config = new HashMap<>();
        this.enabled = Boolean.parseBoolean(context.getConfig("enabled", "true"));
        this.logInputs = Boolean.parseBoolean(context.getConfig("logInputs", "true"));
        this.logOutputs = Boolean.parseBoolean(context.getConfig("logOutputs", "true"));

        LOG.infof("Initialized %s (enabled: %s)", id(), enabled);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        return enabled;
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) {
        String runId = context.token().getExecutionId();

        InferenceRequest request = context.getVariable("request", InferenceRequest.class)
                .orElse(null);

        InferenceResponse response = context.getVariable("response", InferenceResponse.class)
                .orElse(null);

        // Build audit payload
        var auditBuilder = AuditPayload.builder()
                .runId(runId)
                .event("INFERENCE_COMPLETED")
                .level(context.hasError() ? "ERROR" : "INFO")
                .actor(AuditPayload.Actor.system("inference-engine"));

        // Add metadata
        if (request != null) {
            auditBuilder.metadata("model", request.getModel())
                    .metadata("messageCount", request.getMessages().size());

            if (logInputs) {
                auditBuilder.metadata("requestId", request.getRequestId());
            }
        }

        if (response != null) {
            auditBuilder.metadata("tokensUsed", response.getTokensUsed())
                    .metadata("durationMs", response.getDurationMs());

            if (logOutputs) {
                auditBuilder.metadata("contentLength", response.getContent().length());
            }
        }

        if (context.hasError()) {
            context.getError().ifPresent(error -> auditBuilder.metadata("error", error.getMessage())
                    .metadata("errorType", error.getClass().getSimpleName()));
        }

        AuditPayload audit = auditBuilder.build();

        // In a real implementation we would send this to the audit service
        LOG.info("AUDIT: " + audit);
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        this.logInputs = (Boolean) newConfig.getOrDefault("logInputs", true);
        this.logOutputs = (Boolean) newConfig.getOrDefault("logOutputs", true);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return new HashMap<>(config);
    }
}
