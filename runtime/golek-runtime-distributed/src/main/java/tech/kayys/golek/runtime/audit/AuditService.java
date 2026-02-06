package tech.kayys.golek.runtime.audit;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.inference.*;
import tech.kayys.golek.runtime.messaging.AuditEventPublisher;

import java.time.Instant;

/**
 * Central audit service for all inference operations
 */
@ApplicationScoped
public class AuditService {

        private static final Logger LOG = Logger.getLogger(AuditService.class);

        @Inject
        AuditEventPublisher eventPublisher;

        public void logInferenceStart(
                        InferenceRequest request,
                        TenantContext tenantContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("INFERENCE_STARTED")
                                .level("INFO")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("model", request.getModel())
                                .metadata("tenantId", tenantContext.getTenantId())
                                .metadata("streaming", request.isStreaming())
                                .build();

                eventPublisher.publish(audit);
                LOG.debugf("Audit: Inference started - %s", request.getRequestId());
        }

        public void logInferenceComplete(
                        InferenceRequest request,
                        InferenceResponse response,
                        TenantContext tenantContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("INFERENCE_COMPLETED")
                                .level("INFO")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("model", response.getModel())
                                .metadata("tokensUsed", response.getTokensUsed())
                                .metadata("durationMs", response.getDurationMs())
                                .metadata("tenantId", tenantContext.getTenantId())
                                .build();

                eventPublisher.publish(audit);
                LOG.debugf("Audit: Inference completed - %s", request.getRequestId());
        }

        public void logInferenceFailure(
                        InferenceRequest request,
                        Throwable error,
                        TenantContext tenantContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("INFERENCE_FAILED")
                                .level("ERROR")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("model", request.getModel())
                                .metadata("error", error.getClass().getSimpleName())
                                .metadata("message", error.getMessage())
                                .metadata("tenantId", tenantContext.getTenantId())
                                .build();

                eventPublisher.publish(audit);
                LOG.errorf(error, "Audit: Inference failed - %s", request.getRequestId());
        }

        public void logStreamStart(
                        InferenceRequest request,
                        TenantContext tenantContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("STREAM_STARTED")
                                .level("INFO")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("model", request.getModel())
                                .metadata("tenantId", tenantContext.getTenantId())
                                .build();

                eventPublisher.publish(audit);
        }

        public void logStreamComplete(
                        InferenceRequest request,
                        TenantContext tenantContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("STREAM_COMPLETED")
                                .level("INFO")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("tenantId", tenantContext.getTenantId())
                                .build();

                eventPublisher.publish(audit);
        }

        public void logStreamFailure(
                        InferenceRequest request,
                        Throwable error,
                        TenantContext tenantContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("STREAM_FAILED")
                                .level("ERROR")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("error", error.getClass().getSimpleName())
                                .metadata("tenantId", tenantContext.getTenantId())
                                .build();

                eventPublisher.publish(audit);
        }

        public void logCancellation(
                        String requestId,
                        TenantContext tenantContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(requestId)
                                .event("INFERENCE_CANCELLED")
                                .level("WARN")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("tenantId", tenantContext.getTenantId())
                                .build();

                eventPublisher.publish(audit);
        }
}