package tech.kayys.golek.runtime.resource;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.SecurityContext;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.parameters.Parameter;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponse;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.logging.Logger;
import org.jboss.resteasy.reactive.RestStreamElementType;

import java.util.Map;
import java.util.UUID;

// Import the required classes
import tech.kayys.golek.api.inference.*;
import tech.kayys.golek.runtime.service.PlatformInferenceService;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.runtime.security.TenantContextResolver;

/**
 * Main inference REST endpoint.
 * Handles synchronous and streaming inference requests.
 */
@Path("/v1/inference")
@Tag(name = "Inference", description = "LLM inference operations")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class InferenceResource {

        private static final Logger LOG = Logger.getLogger(InferenceResource.class);

        @Inject
        PlatformInferenceService inferenceService;

        @Inject
        TenantContextResolver tenantResolver;

        /**
         * Synchronous inference endpoint
         */
        @POST
        @Path("/infer")
        @Operation(summary = "Execute synchronous inference")
        @APIResponse(responseCode = "200", description = "Inference completed successfully")
        @APIResponse(responseCode = "400", description = "Invalid request")
        @APIResponse(responseCode = "401", description = "Unauthorized")
        @APIResponse(responseCode = "429", description = "Rate limit exceeded")
        @APIResponse(responseCode = "500", description = "Internal server error")
        public Uni<Response> infer(
                        @Valid @NotNull InferenceRequest request,
                        @Context SecurityContext securityContext) {
                String requestId = request.getRequestId() != null
                                ? request.getRequestId()
                                : UUID.randomUUID().toString();

                LOG.infof("Inference request received: %s, model: %s",
                                requestId, request.getModel());

                TenantContext tenantContext = tenantResolver.resolve(securityContext);

                return inferenceService.infer(request, tenantContext)
                                .map(response -> Response.ok(response).build())
                                .onFailure().recoverWithItem(error -> {
                                        LOG.errorf(error, "Inference failed for request: %s", requestId);
                                        return buildErrorResponse(error, requestId);
                                });
        }

        /**
         * Streaming inference endpoint
         */
        @POST
        @Path("/stream")
        @Operation(summary = "Execute streaming inference")
        @APIResponse(responseCode = "200", description = "Stream started successfully")
        @Produces(MediaType.SERVER_SENT_EVENTS)
        @RestStreamElementType(MediaType.APPLICATION_JSON)
        public Uni<Multi<StreamChunk>> stream(
                        @Valid @NotNull InferenceRequest request,
                        @Context SecurityContext securityContext) {
                String requestId = request.getRequestId() != null
                                ? request.getRequestId()
                                : UUID.randomUUID().toString();

                LOG.infof("Stream request received: %s, model: %s",
                                requestId, request.getModel());

                TenantContext tenantContext = tenantResolver.resolve(securityContext);

                return Uni.createFrom().item(() -> inferenceService.stream(request, tenantContext));
        }

        /**
         * Batch inference endpoint
         */
        @POST
        @Path("/batch")
        @Operation(summary = "Execute batch inference")
        @APIResponse(responseCode = "202", description = "Batch accepted for processing")
        public Uni<Response> batch(
                        @Valid @NotNull BatchInferenceRequest batchRequest,
                        @Context SecurityContext securityContext) {
                LOG.infof("Batch request received: %d requests",
                                batchRequest.getRequests().size());

                TenantContext tenantContext = tenantResolver.resolve(securityContext);

                return inferenceService.batchInfer(batchRequest, tenantContext)
                                .map(batchId -> Response
                                                .accepted()
                                                .entity(Map.of(
                                                                "batchId", batchId,
                                                                "status", "ACCEPTED",
                                                                "totalRequests", batchRequest.getRequests().size()))
                                                .build());
        }

        /**
         * Get batch status
         */
        @GET
        @Path("/batch/{batchId}")
        @Operation(summary = "Get batch inference status")
        public Uni<Response> getBatchStatus(
                        @Parameter(description = "Batch ID") @PathParam("batchId") String batchId,
                        @Context SecurityContext securityContext) {
                TenantContext tenantContext = tenantResolver.resolve(securityContext);

                return inferenceService.getBatchStatus(batchId, tenantContext)
                                .map(status -> Response.ok(status).build())
                                .onFailure()
                                .recoverWithItem(error -> Response.status(Response.Status.NOT_FOUND).build());
        }

        /**
         * Cancel inference request
         */
        @DELETE
        @Path("/{requestId}")
        @Operation(summary = "Cancel inference request")
        public Uni<Response> cancel(
                        @Parameter(description = "Request ID") @PathParam("requestId") String requestId,
                        @Context SecurityContext securityContext) {
                LOG.infof("Cancel request: %s", requestId);

                TenantContext tenantContext = tenantResolver.resolve(securityContext);

                return inferenceService.cancel(requestId, tenantContext)
                                .map(cancelled -> Response
                                                .ok(Map.of(
                                                                "requestId", requestId,
                                                                "cancelled", cancelled))
                                                .build());
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