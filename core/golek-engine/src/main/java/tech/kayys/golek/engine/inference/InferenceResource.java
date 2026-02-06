package tech.kayys.golek.engine.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.annotation.security.RolesAllowed;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.SecurityContext;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.faulttolerance.*;
import org.eclipse.microprofile.metrics.annotation.Counted;
import org.eclipse.microprofile.metrics.annotation.Timed;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.media.Content;
import org.eclipse.microprofile.openapi.annotations.media.Schema;
import org.eclipse.microprofile.openapi.annotations.parameters.Parameter;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponse;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponses;
import org.eclipse.microprofile.openapi.annotations.security.SecurityRequirement;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;

import lombok.extern.slf4j.Slf4j;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.error.ErrorCode;

import java.time.temporal.ChronoUnit;
import java.util.UUID;

/**
 * REST API endpoint for ML model inference.
 * 
 * <p>
 * Features:
 * <ul>
 * <li>Synchronous and asynchronous inference</li>
 * <li>Streaming for generative models</li>
 * <li>JWT-based authentication</li>
 * <li>Tenant isolation</li>
 * <li>Rate limiting</li>
 * <li>Circuit breaker & retry logic</li>
 * <li>Comprehensive metrics</li>
 * <li>OpenAPI documentation</li>
 * </ul>
 * 
 * @author bhangun
 * @since 1.0.0
 */
@Path("/v1/infer")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Inference", description = "Model inference operations")
@Slf4j
public class InferenceResource {

        @Inject
        InferenceService inferenceService;

        @Inject
        InferenceMetrics metrics;

        @ConfigProperty(name = "wayang.multitenancy.enabled", defaultValue = "false")
        boolean multitenancyEnabled;

        /**
         * Execute synchronous inference.
         * 
         * @param tenantId        Tenant identifier from header
         * @param requestId       Optional request ID for tracing
         * @param request         Inference request with model ID and inputs
         * @param securityContext Security context with authenticated user
         * @return Inference response with outputs and metrics
         */
        @POST
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Run inference", description = "Execute synchronous inference on the specified model")
        @SecurityRequirement(name = "bearer-jwt")
        @APIResponses({
                        @APIResponse(responseCode = "200", description = "Inference successful", content = @Content(schema = @Schema(implementation = InferenceResponse.class))),
                        @APIResponse(responseCode = "400", description = "Invalid request", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
                        @APIResponse(responseCode = "401", description = "Unauthorized"),
                        @APIResponse(responseCode = "403", description = "Forbidden - insufficient permissions"),
                        @APIResponse(responseCode = "404", description = "Model not found"),
                        @APIResponse(responseCode = "429", description = "Rate limit exceeded"),
                        @APIResponse(responseCode = "503", description = "Service unavailable"),
                        @APIResponse(responseCode = "504", description = "Request timeout")
        })
        @Counted(name = "inference.requests.total", description = "Total inference requests")
        @Timed(name = "inference.request.duration", description = "Inference request duration")
        @Timeout(value = 30, unit = ChronoUnit.SECONDS)
        @Retry(maxRetries = 2, delay = 100, delayUnit = ChronoUnit.MILLIS)
        @CircuitBreaker(requestVolumeThreshold = 20, failureRatio = 0.5, delay = 5000, successThreshold = 2)
        @Bulkhead(value = 100, waitingTaskQueue = 50)
        public Uni<Response> infer(
                        @Parameter(description = "Tenant identifier") @HeaderParam("X-Tenant-ID") String tenantId,

                        @Parameter(description = "Request ID for tracing") @HeaderParam("X-Request-ID") String requestId,

                        @Parameter(description = "Inference request payload", required = true) @Valid @NotNull InferenceRequest request,

                        @Context SecurityContext securityContext) {
                String effectiveTenantId = resolveTenantId(tenantId);

                // 1. Prepare Request
                InferenceRequest effectiveRequest = InferenceRequest.builder()
                                .requestId(requestId)
                                .model(request.getModel())
                                .messages(request.getMessages())
                                .tools(request.getTools())
                                .toolChoice(request.getToolChoice())
                                .parameters(request.getParameters())
                                .parameter("tenantId", effectiveTenantId)
                                .streaming(false)
                                .priority(request.getPriority())
                                .build();

                log.info("Inference request received: requestId={}, tenantId={}, model={}",
                                requestId, effectiveTenantId, request.getModel());

                // Execute inference asynchronously
                return inferenceService.inferAsync(effectiveRequest)
                                .map(response -> {
                                        log.info("Inference completed: requestId={}, durationMs={}, model={}",
                                                        requestId, response.getDurationMs(), response.getModel());

                                        metrics.recordSuccess(effectiveTenantId, request.getModel(),
                                                        "unified", response.getDurationMs());

                                        return Response.ok(response).build();
                                })
                                .onFailure().recoverWithItem(failure -> {
                                        log.error("Inference failed: requestId={}", requestId, failure);

                                        metrics.recordFailure(effectiveTenantId, request.getModel(),
                                                        failure.getClass().getSimpleName());

                                        return handleError(failure, requestId);
                                });
        }

        /**
         * Execute asynchronous inference and return immediately.
         * Client can poll for results using returned job ID.
         */
        @POST
        @Path("/async")
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Submit async inference job", description = "Submit inference request for asynchronous processing")
        @SecurityRequirement(name = "bearer-jwt")
        public Uni<Response> inferAsync(
                        @HeaderParam("X-Tenant-ID") String tenantId,
                        @HeaderParam("X-Request-ID") String requestId,
                        @Valid @NotNull InferenceRequest request,
                        @Context SecurityContext securityContext) {
                String effectiveTenantId = resolveTenantId(tenantId);

                if (requestId == null) {
                        requestId = UUID.randomUUID().toString();
                }

                InferenceRequest effectiveRequest = InferenceRequest.builder()
                                .requestId(requestId)
                                .model(request.getModel())
                                .messages(request.getMessages())
                                .tools(request.getTools())
                                .toolChoice(request.getToolChoice())
                                .parameters(request.getParameters())
                                .parameter("tenantId", effectiveTenantId)
                                .streaming(false)
                                .priority(request.getPriority())
                                .build();

                return inferenceService.submitAsyncJob(effectiveRequest)
                                .map(jobId -> Response.accepted()
                                                .entity(new AsyncJobResponse(jobId, requestId))
                                                .build());
        }

        /**
         * Stream inference results for generative models.
         * Returns Server-Sent Events (SSE) stream.
         */
        @POST
        @Path("/stream")
        @Produces(MediaType.SERVER_SENT_EVENTS)
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Stream inference results", description = "Execute streaming inference (e.g., for LLMs token generation)")
        @SecurityRequirement(name = "bearer-jwt")
        public Multi<StreamingInferenceChunk> inferStream(
                        @HeaderParam("X-Tenant-ID") String tenantId,
                        @HeaderParam("X-Request-ID") String requestId,
                        @Valid @NotNull InferenceRequest request,
                        @Context SecurityContext securityContext) {
                String effectiveTenantId = resolveTenantId(tenantId);

                if (requestId == null) {
                        requestId = UUID.randomUUID().toString();
                }

                InferenceRequest effectiveRequest = InferenceRequest.builder()
                                .requestId(requestId)
                                .model(request.getModel())
                                .messages(request.getMessages())
                                .tools(request.getTools())
                                .toolChoice(request.getToolChoice())
                                .parameters(request.getParameters())
                                .parameter("tenantId", effectiveTenantId)
                                .streaming(true)
                                .priority(request.getPriority())
                                .build();

                log.info("Streaming inference request: requestId={}, model={}",
                                requestId, request.getModel());

                return inferenceService.inferStream(effectiveRequest)
                                .onFailure()
                                .invoke(failure -> log.error("Streaming inference failed: requestId={}", requestId,
                                                failure));
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
                        @HeaderParam("X-Tenant-ID") String tenantId) {
                String effectiveTenantId = resolveTenantId(tenantId);

                return inferenceService.getJobStatus(jobId, effectiveTenantId)
                                .map(status -> Response.ok(status).build())
                                .onFailure().recoverWithItem(failure -> handleError(failure, null));
        }

        /**
         * Batch inference - process multiple requests in a single call.
         */
        @POST
        @Path("/batch")
        @RolesAllowed({ "USER", "ADMIN" })
        @Operation(summary = "Batch inference", description = "Process multiple inference requests in a single batch")
        @SecurityRequirement(name = "bearer-jwt")
        public Uni<Response> batchInfer(
                        @HeaderParam("X-Tenant-ID") String tenantId,
                        @Valid @NotNull BatchInferenceRequest batchRequest,
                        @Context SecurityContext securityContext) {
                String effectiveTenantId = resolveTenantId(tenantId);

                InferenceService.BatchInferenceRequest reactiveBatchRequest = new InferenceService.BatchInferenceRequest(
                                batchRequest.requests(), batchRequest.maxConcurrent());

                return inferenceService.batchInfer(reactiveBatchRequest, effectiveTenantId)
                                .map(responses -> Response.ok(responses).build())
                                .onFailure().recoverWithItem(failure -> handleError(failure, null));
        }

        private String resolveTenantId(String tenantId) {
                if (!multitenancyEnabled) {
                        return "default";
                }

                if (tenantId == null || tenantId.trim().isEmpty()) {
                        throw new BadRequestException("Tenant ID header (X-Tenant-ID) is required");
                }

                return tenantId;
        }

        /**
         * Handle errors and convert to appropriate HTTP responses.
         */
        private Response handleError(Throwable failure, String requestId) {
                if (failure instanceof InferenceException ie) {
                        ErrorResponse errorResponse = ErrorResponse.fromException(ie, requestId);

                        // Specific handling for common errors
                        return Response.status(ie.getHttpStatusCode())
                                        .entity(errorResponse)
                                        .build();
                }

                // Unexpected error - return 500
                log.error("Unexpected error during inference", failure);
                return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                                .entity(ErrorResponse.builder()
                                                .errorCode(ErrorCode.INTERNAL_ERROR.getCode())
                                                .message("Internal server error")
                                                .httpStatus(500)
                                                .requestId(requestId)
                                                .build())
                                .build();
        }

        /**
         * Response model for async job submission.
         */
        public record AsyncJobResponse(String jobId, String requestId) {
        }

        /**
         * Streaming inference chunk.
         */
        public record StreamingInferenceChunk(
                        String requestId,
                        int sequenceNumber,
                        String token,
                        boolean isComplete,
                        Long latencyMs) {
        }

        /**
         * Batch inference request.
         */
        public record BatchInferenceRequest(
                        java.util.List<InferenceRequest> requests,
                        Integer maxConcurrent) {
        }
}
