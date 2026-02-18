package tech.kayys.gollek.observability;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.AuditPayload;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import org.jboss.logging.Logger;

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
                        RequestContext requestContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("INFERENCE_STARTED")
                                .level("INFO")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("model", request.getModel())
                                .metadata("requestId", requestContext.requestId())
                                .metadata("streaming", request.isStreaming())
                                .build();

                eventPublisher.publish(audit);
                LOG.debugf("Audit: Inference started - %s", request.getRequestId());
        }

        public void logInferenceComplete(
                        InferenceRequest request,
                        InferenceResponse response,
                        RequestContext requestContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("INFERENCE_COMPLETED")
                                .level("INFO")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("model", response.getModel())
                                .metadata("tokensUsed", response.getTokensUsed())
                                .metadata("durationMs", response.getDurationMs())
                                .metadata("requestId", requestContext.requestId())
                                .build();

                eventPublisher.publish(audit);
                LOG.debugf("Audit: Inference completed - %s", request.getRequestId());
        }

        public void logInferenceFailure(
                        InferenceRequest request,
                        Throwable error,
                        RequestContext requestContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("INFERENCE_FAILED")
                                .level("ERROR")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("model", request.getModel())
                                .metadata("error", error.getClass().getSimpleName())
                                .metadata("message", error.getMessage())
                                .metadata("requestId", requestContext.requestId())
                                .build();

                eventPublisher.publish(audit);
                LOG.errorf(error, "Audit: Inference failed - %s", request.getRequestId());
        }

        public void logStreamStart(
                        InferenceRequest request,
                        RequestContext requestContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("STREAM_STARTED")
                                .level("INFO")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("model", request.getModel())
                                .metadata("requestId", requestContext.requestId())
                                .build();

                eventPublisher.publish(audit);
        }

        public void logStreamComplete(
                        InferenceRequest request,
                        RequestContext requestContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("STREAM_COMPLETED")
                                .level("INFO")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("requestId", requestContext.requestId())
                                .build();

                eventPublisher.publish(audit);
        }

        public void logStreamFailure(
                        InferenceRequest request,
                        Throwable error,
                        RequestContext requestContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(request.getRequestId())
                                .event("STREAM_FAILED")
                                .level("ERROR")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("error", error.getClass().getSimpleName())
                                .metadata("requestId", requestContext.requestId())
                                .build();

                eventPublisher.publish(audit);
        }

        public void logCancellation(
                        String requestId,
                        RequestContext requestContext) {
                AuditPayload audit = AuditPayload.builder()
                                .runId(requestId)
                                .event("INFERENCE_CANCELLED")
                                .level("WARN")
                                .actor(AuditPayload.Actor.system("inference-platform"))
                                .metadata("requestId", requestContext.requestId())
                                .build();

                eventPublisher.publish(audit);
        }
}