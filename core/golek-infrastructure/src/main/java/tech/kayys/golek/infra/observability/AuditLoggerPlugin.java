package tech.kayys.golek.infra.observability;

import tech.kayys.golek.spi.AuditPayload;
import tech.kayys.golek.spi.inference.*;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.core.plugin.InferencePhasePlugin;
import tech.kayys.golek.spi.plugin.PluginException;
import tech.kayys.golek.spi.exception.ProviderException;
import java.util.Map;
import java.util.HashMap;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.validation.ValidationException;

/**
 * Plugin that logs audit events for all inference requests.
 * Executes in AUDIT phase after inference completion.
 */
@ApplicationScoped
public class AuditLoggerPlugin implements InferencePhasePlugin {

    @Inject
    AuditEventPublisher publisher;

    @Inject
    ProvenanceStore provenanceStore;

    @Override
    public String id() {
        return "tech.kayys.golek.audit.logger";
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.AUDIT;
    }

    @Override
    public int order() {
        return 10;
    }

    @Override
    public void execute(
            ExecutionContext context,
            EngineContext engine) throws PluginException {

        AuditPayload audit = AuditPayload.builder()
                .runId(context.token().getRequestId())
                .event(determineEvent(context))
                .level(determineLevel(context))
                .tag("inference")
                .tag(context.tenantContext().getTenantId().value())
                .metadata("modelId", context.getVariable("request", InferenceRequest.class)
                        .map(InferenceRequest::getModel).orElse("unknown"))
                .metadata("providerId", getProviderId(context))
                .metadata("duration", getDuration(context))
                .metadata("tokensUsed", getTokensUsed(context))
                .contextSnapshot(buildSnapshot(context))
                .build();

        // Store and publish asynchronously (fire and forget for now, or handle
        // blocking)
        provenanceStore.store(audit)
                .chain(() -> publisher.publish(audit))
                .subscribe().with(
                        v -> {
                        },
                        e -> {
                        } // Log publisher error
                );
    }

    private Map<String, Object> config = new HashMap<>();

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.copyOf(config);
    }

    private String determineEvent(ExecutionContext context) {
        if (context.hasError()) {
            return "INFERENCE_FAILED";
        } else {
            return "INFERENCE_COMPLETED";
        }
    }

    private String determineLevel(ExecutionContext context) {
        if (context.hasError()) {
            Throwable error = context.getError().get();
            if (error instanceof ValidationException) {
                return "WARN";
            } else if (error instanceof ProviderException) {
                return "ERROR";
            } else {
                return "CRITICAL";
            }
        } else {
            return "INFO";
        }
    }

    private String getProviderId(ExecutionContext context) {
        return context.getVariable("providerId", String.class)
                .orElse("unknown");
    }

    private long getDuration(ExecutionContext context) {
        return context.getVariable("durationMs", Long.class)
                .orElse(0L);
    }

    private int getTokensUsed(ExecutionContext context) {
        return context.getVariable("response", InferenceResponse.class)
                .map(InferenceResponse::getTokensUsed)
                .orElse(0);
    }

    private Map<String, Object> buildSnapshot(ExecutionContext context) {
        Map<String, Object> snapshot = new HashMap<>();
        snapshot.put("requestId", context.token().getRequestId());
        snapshot.put("tenantId", context.tenantContext().getTenantId().value());

        context.getVariable("request", InferenceRequest.class)
                .ifPresent(req -> snapshot.put("model", req.getModel()));

        context.getVariable("response", InferenceResponse.class)
                .ifPresentOrElse(
                        resp -> {
                            snapshot.put("responseStatus", "success");
                            snapshot.put("tokensUsed", resp.getTokensUsed());
                        },
                        () -> {
                            if (context.hasError()) {
                                snapshot.put("responseStatus", "error");
                                snapshot.put("errorType", context.getError().get().getClass().getSimpleName());
                            }
                        });

        return snapshot;
    }
}