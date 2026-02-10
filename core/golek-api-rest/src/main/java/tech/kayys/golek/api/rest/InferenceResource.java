package tech.kayys.golek.api.rest;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.annotation.security.RolesAllowed;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.ws.rs.*;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.SecurityContext;

import org.eclipse.microprofile.faulttolerance.*;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.media.Content;
import org.eclipse.microprofile.openapi.annotations.media.Schema;
import org.eclipse.microprofile.openapi.annotations.parameters.Parameter;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponse;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponses;
import org.eclipse.microprofile.openapi.annotations.security.SecurityRequirement;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.resteasy.reactive.RestStreamElementType;

import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.inference.StreamingInferenceChunk;
import tech.kayys.wayang.error.ErrorResponse;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.api.dto.AsyncJobResponse;
import tech.kayys.golek.api.dto.InferenceResponseDTO;
import tech.kayys.golek.api.dto.StreamChunkDTO;
import tech.kayys.golek.spi.inference.BatchInferenceRequest;
import tech.kayys.golek.engine.model.PlatformInferenceService;
import tech.kayys.golek.engine.inference.InferenceMetrics;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.error.ErrorPayload;
import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.spi.exception.InferenceException;
import tech.kayys.wayang.tenant.TenantContextResolver;

import java.time.temporal.ChronoUnit;
import java.util.*;

/**
 * REST API endpoint for ML model inference.
 */
@Path("/v1/infer")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Inference", description = "Model inference operations")
public class InferenceResource {

        private static final Logger LOG = Logger.getLogger(InferenceResource.class);

        @Inject
        InferenceMetrics inferenceMetrics;

        @Inject
        PlatformInferenceService platformInferenceService;

        @Inject
        TenantContextResolver tenantResolver;

        /**
         * Execute synchronous inference.
         */
        @POST
        @Path("/completions")
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Run inference", description = "Execute synchronous inference on the specified model")
        @SecurityRequirement(name = "bearer-jwt")
        @APIResponses({
                        @APIResponse(responseCode = "200", description = "Inference successful", content = @Content(schema = @Schema(implementation = InferenceResponseDTO.class))),
                        @APIResponse(responseCode = "400", description = "Invalid request", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
                        @APIResponse(responseCode = "401", description = "Unauthorized"),
                        @APIResponse(responseCode = "429", description = "Rate limit exceeded"),
                        @APIResponse(responseCode = "500", description = "Internal server error")
        })
        @Timeout(value = 30, unit = ChronoUnit.SECONDS)
        @Retry(maxRetries = 2, delay = 100, delayUnit = ChronoUnit.MILLIS)
        @CircuitBreaker(requestVolumeThreshold = 20, failureRatio = 0.5, delay = 5000, successThreshold = 2)
        @Bulkhead(value = 100, waitingTaskQueue = 50)
        public Uni<Response> infer(
                        @Parameter(description = "Request ID for tracing") @HeaderParam("X-Request-ID") String requestId,
                        @Valid @NotNull InferenceRequest request,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                final String finalRequestId = requestId != null ? requestId : UUID.randomUUID().toString();
                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);
                String effectiveTenantId = tenantContext.getTenantId().value();

                InferenceRequest effectiveRequest = InferenceRequest.builder()
                                .requestId(finalRequestId)
                                .tenantId(effectiveTenantId)
                                .model(request.getModel())
                                .messages(request.getMessages())
                                .tools(request.getTools())
                                .toolChoice(request.getToolChoice())
                                .parameters(request.getParameters())
                                .streaming(false)
                                .priority(request.getPriority())
                                .build();

                LOG.infof("Inference request received: requestId=%s, tenantId=%s, model=%s",
                                finalRequestId, effectiveTenantId, request.getModel());

                return platformInferenceService.infer(effectiveRequest, tenantContext)
                                .map(response -> {
                                        LOG.infof("Inference completed: requestId=%s, durationMs=%d, model=%s",
                                                        finalRequestId, response.getDurationMs(), response.getModel());

                                        inferenceMetrics.recordSuccess(effectiveTenantId, request.getModel(),
                                                        "unified", response.getDurationMs());

                                        return Response.ok(InferenceResponseDTO.from(response)).build();
                                })
                                .onFailure().recoverWithItem(failure -> {
                                        LOG.errorf(failure, "Inference failed: requestId=%s", finalRequestId);

                                        inferenceMetrics.recordFailure(effectiveTenantId, request.getModel(),
                                                        failure.getClass().getSimpleName());

                                        return buildErrorResponse(failure, finalRequestId);
                                });
        }

        /**
         * Execute asynchronous inference.
         */
        @POST
        @Path("/async")
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Submit async inference job")
        @SecurityRequirement(name = "bearer-jwt")
        public Uni<Response> inferAsync(
                        @HeaderParam("X-Request-ID") String requestId,
                        @Valid @NotNull InferenceRequest request,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);
                String effectiveTenantId = tenantContext.getTenantId().value();
                final String finalRequestId = requestId != null ? requestId : UUID.randomUUID().toString();

                InferenceRequest effectiveRequest = InferenceRequest.builder()
                                .requestId(finalRequestId)
                                .tenantId(effectiveTenantId)
                                .model(request.getModel())
                                .messages(request.getMessages())
                                .tools(request.getTools())
                                .toolChoice(request.getToolChoice())
                                .parameters(request.getParameters())
                                .streaming(false)
                                .priority(request.getPriority())
                                .build();

                return platformInferenceService.submitAsyncJob(effectiveRequest, tenantContext)
                                .map(jobId -> Response.accepted()
                                                .entity(new AsyncJobResponse(jobId, finalRequestId))
                                                .build());
        }

        /**
         * Stream inference results.
         */
        @POST
        @Path("/stream")
        @Produces(MediaType.SERVER_SENT_EVENTS)
        @RestStreamElementType(MediaType.APPLICATION_JSON)
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Stream inference results")
        @SecurityRequirement(name = "bearer-jwt")
        public Multi<StreamChunkDTO> inferStream(
                        @HeaderParam("X-Request-ID") String requestId,
                        @Valid @NotNull InferenceRequest request,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                final String finalRequestId = requestId != null ? requestId : UUID.randomUUID().toString();
                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);
                String effectiveTenantId = tenantContext.getTenantId().value();

                InferenceRequest effectiveRequest = InferenceRequest.builder()
                                .requestId(finalRequestId)
                                .tenantId(effectiveTenantId)
                                .model(request.getModel())
                                .messages(request.getMessages())
                                .tools(request.getTools())
                                .toolChoice(request.getToolChoice())
                                .parameters(request.getParameters())
                                .streaming(true)
                                .priority(request.getPriority())
                                .build();

                LOG.infof("Streaming inference request: requestId=%s, model=%s",
                                finalRequestId, request.getModel());

                return platformInferenceService.stream(effectiveRequest, tenantContext)
                                .map(chunk -> new StreamChunkDTO(chunk.sequenceNumber(), chunk.token(),
                                                chunk.isComplete()));
        }

        /**
         * Get status of async inference job.
         */
        @GET
        @Path("/async/{jobId}")
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Get async job status")
        @SecurityRequirement(name = "bearer-jwt")
        public Uni<Response> getAsyncJobStatus(
                        @PathParam("jobId") String jobId,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);

                return platformInferenceService.getBatchStatus(jobId, tenantContext)
                                .map(status -> Response.ok(status).build());
        }

        /**
         * Batch inference.
         */
        @POST
        @Path("/batch")
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Batch inference")
        @SecurityRequirement(name = "bearer-jwt")
        public Uni<Response> batchInfer(
                        @Valid @NotNull BatchInferenceRequest batchRequest,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);

                return platformInferenceService.batchInfer(batchRequest, tenantContext)
                                .map(batchId -> Response.accepted()
                                                .entity(Map.of("batchId", batchId))
                                                .build());
        }

        /**
         * Cancel inference request.
         */
        @DELETE
        @Path("/{requestId}")
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Cancel inference request")
        @SecurityRequirement(name = "bearer-jwt")
        public Uni<Response> cancel(
                        @Parameter(description = "Request ID") @PathParam("requestId") String requestId,
                        @Context SecurityContext securityContext) {
                LOG.infof("Cancel request: %s", requestId);
                TenantContext tenantContext = tenantResolver.resolve(securityContext);

                return platformInferenceService.cancel(requestId, tenantContext)
                                .map(cancelled -> Response
                                                .ok(Map.of(
                                                                "requestId", requestId,
                                                                "cancelled", cancelled))
                                                .build());
        }

        private TenantContext resolveTenantContext(SecurityContext securityContext,
                        ContainerRequestContext requestContext) {
                Object ctx = requestContext != null ? requestContext.getProperty("tenantContext") : null;
                if (ctx instanceof TenantContext tenantContext) {
                        return tenantContext;
                }
                return tenantResolver.resolve(securityContext);
        }

        private Response buildErrorResponse(Throwable error, String requestId) {
                ErrorPayload errorPayload = ErrorPayload.builder()
                                .type(error.getClass().getSimpleName())
                                .message(error.getMessage())
                                .originNode("inference-platform")
                                .originRunId(requestId)
                                .retryable(isRetryable(error))
                                .suggestedAction(determineSuggestedAction(error))
                                .build();

                Response.Status status = determineHttpStatus(error);

                return Response
                                .status(status)
                                .entity(errorPayload)
                                .build();
        }

        private boolean isRetryable(Throwable error) {
                String className = error.getClass().getName();
                return !className.contains("Validation") &&
                                !className.contains("Authorization") &&
                                !className.contains("Quota");
        }

        private String determineSuggestedAction(Throwable error) {
                String className = error.getClass().getName();
                if (className.contains("Quota")) {
                        return "escalate";
                } else if (className.contains("Provider")) {
                        return "retry";
                } else if (className.contains("Validation")) {
                        return "human_review";
                }
                return "fallback";
        }

        private Response.Status determineHttpStatus(Throwable error) {
                String className = error.getClass().getName();
                if (className.contains("Validation")) {
                        return Response.Status.BAD_REQUEST;
                } else if (className.contains("Authorization") ||
                                className.contains("Authentication")) {
                        return Response.Status.UNAUTHORIZED;
                } else if (className.contains("Quota") ||
                                className.contains("RateLimit")) {
                        return Response.Status.TOO_MANY_REQUESTS;
                } else if (className.contains("NotFound")) {
                        return Response.Status.NOT_FOUND;
                }
                return Response.Status.INTERNAL_SERVER_ERROR;
        }
}
