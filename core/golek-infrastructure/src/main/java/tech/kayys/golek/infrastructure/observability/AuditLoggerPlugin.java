package tech.kayys.golek.inference.infrastructure.observability;

import tech.kayys.golek.inference.kernel.plugin.*;
import tech.kayys.golek.inference.kernel.engine.*;
import tech.kayys.golek.inference.kernel.pipeline.*;
import io.smallrye.mutiny.Uni;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

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
    public PluginDescriptor descriptor() {
        return new PluginDescriptor(
            id(),
            "Audit Logger Plugin",
            "1.0.0",
            PluginType.OBSERVABILITY,
            Set.of(PluginCapability.ASYNC_EXECUTION),
            SemanticVersion.of("1.0.0"),
            SemanticVersion.of("2.0.0"),
            Map.of("category", "audit")
        );
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.AUDIT;
    }

    @Override
    public int order() {
        return 10; // Execute early in audit phase
    }

    @Override
    public Uni<Void> execute(
        InferenceContext context,
        EngineContext engine
    ) {
        AuditPayload audit = context.auditBuilder()
            .event(determineEvent(context))
            .level(determineLevel(context))
            .tag("inference")
            .tag(context.tenantContext().tenantId().value())
            .metadata("modelId", context.request().model())
            .metadata("providerId", getProviderId(context))
            .metadata("duration", getDuration(context))
            .metadata("tokensUsed", getTokensUsed(context))
            .contextSnapshot(buildSnapshot(context))
            .build();

        // Store in provenance
        return provenanceStore.store(audit)
            // Publish to event bus (Kafka)
            .chain(() -> publisher.publish(audit));
    }

    private String determineEvent(InferenceContext context) {
        if (context.hasError()) {
            return "INFERENCE_FAILED";
        } else {
            return "INFERENCE_COMPLETED";
        }
    }

    private String determineLevel(InferenceContext context) {
        if (context.hasError()) {
            Throwable error = context.error().get();
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

    private String getProviderId(InferenceContext context) {
        return context.getAttribute("providerId", String.class)
            .orElse("unknown");
    }

    private long getDuration(InferenceContext context) {
        return context.getAttribute("durationMs", Long.class)
            .orElse(0L);
    }

    private int getTokensUsed(InferenceContext context) {
        return context.response()
            .map(InferenceResponse::tokensUsed)
            .orElse(0);
    }

    private Map<String, Object> buildSnapshot(InferenceContext context) {
        Map<String, Object> snapshot = new HashMap<>();
        snapshot.put("requestId", context.requestId());
        snapshot.put("tenantId", context.tenantContext().tenantId().value());
        snapshot.put("model", context.request().model());
        
        // Redact sensitive data
        if (context.response().isPresent()) {
            snapshot.put("responseStatus", "success");
            snapshot.put("tokensUsed", context.response().get().tokensUsed());
        } else if (context.hasError()) {
            snapshot.put("responseStatus", "error");
            snapshot.put("errorType", context.error().get().getClass().getSimpleName());
        }
        
        return snapshot;
    }
}