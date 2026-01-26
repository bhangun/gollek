package tech.kayys.golek.core.inference;

import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.*;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.media.Content;
import org.eclipse.microprofile.openapi.annotations.media.Schema;
import org.eclipse.microprofile.openapi.annotations.parameters.Parameter;
import org.eclipse.microprofile.openapi.annotations.responses.APIResponse;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.logging.Logger;
import org.jboss.resteasy.reactive.RestStreamElementType;
import *;
import tech.kayys.wayang.inference.core.service.InferenceOrchestrator;
import tech.kayys.wayang.inference.core.service.ModelRouterService;
import tech.kayys.wayang.inference.platform.rest.dto.*;
import tech.kayys.wayang.inference.platform.rest.filter.TenantContext;

import java.util.UUID;

/**
 * Main inference API endpoints
 */
@Path("/v1/inference")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Inference", description = "Model inference operations")
public class InferenceResource {

        private static final Logger LOG = Logger.getLogger(InferenceResource.class);

        @Inject
        InferenceOrchestrator orchestrator;

        @Inject
        ModelRouterService modelRouter;

        @Context
        SecurityContext securityContext;

        /**
         * Synchronous inference
         */
        @POST
        @Path("/completions")
        @Operation(summary = "Create completion", description = "Generate text completion from a model")
        @APIResponse(responseCode = "200", description = "Successful completion", content = @Content(schema = @Schema(implementation = InferenceResponseDTO.class)))
        @APIResponse(responseCode = "400", description = "Invalid request")
        @APIResponse(responseCode = "429", description = "Rate limit exceeded")
        @APIResponse(responseCode = "500", description = "Internal server error")
        public Uni<Response> createCompletion(
                        @Valid InferenceRequestDTO request,
                        @HeaderParam("X-Tenant-ID") String tenantId) {
                LOG.debugf("Inference request for model: %s", request.model());

                return Uni.createFrom().item(() -> {
                        // Build request
                        InferenceRequest inferenceRequest = InferenceRequest.builder()
                                        .requestId(UUID.randomUUID().toString())
                                        .model(request.model())
                                        .messages(request.messages().stream()
                                                        .map(m -> new Message(
                                                                        Message.Role.valueOf(m.role().toUpperCase()),
                                                                        m.content()))
                                                        .toList())
                                        .parameters(request.parameters())
                                        .streaming(false)
                                        .timeout(request.timeout())
                                        .priority(request.priority())
                                        .build();

                        // Build tenant context
                        TenantContext context = buildTenantContext(tenantId);

                        return new RequestContext(inferenceRequest, context);
                })
                                .onItem().transformToUni(ctx -> orchestrator.execute(
                                                ctx.request().getModel(),
                                                ctx.request(),
                                                ctx.tenantContext()))
                                .onItem()
                                .transform(response -> Response.ok(InferenceResponseDTO.from(response)).build())
                                .onFailure().recoverWithItem(error -> {
                                        LOG.error("Inference failed", error);
                                        return Response
                                                        .status(Response.Status.INTERNAL_SERVER_ERROR)
                                                        .entity(ErrorResponseDTO.from(error))
                                                        .build();
                                });
        }

        /**
         * Streaming inference
         */
        @POST
        @Path("/completions/stream")
        @Produces(MediaType.SERVER_SENT_EVENTS)
        @RestStreamElementType(MediaType.APPLICATION_JSON)
        @Operation(summary = "Stream completion", description = "Stream text completion from a model")
        public Multi<StreamChunkDTO> streamCompletion(
                        @Valid InferenceRequestDTO request,
                        @HeaderParam("X-Tenant-ID") String tenantId) {
                LOG.debugf("Streaming inference for model: %s", request.model());

                // Build streaming request
                InferenceRequest inferenceRequest = InferenceRequest.builder()
                                .requestId(UUID.randomUUID().toString())
                                .model(request.model())
                                .messages(request.messages().stream()
                                                .map(m -> new Message(
                                                                Message.Role.valueOf(m.role().toUpperCase()),
                                                                m.content()))
                                                .toList())
                                .parameters(request.parameters())
                                .streaming(true)
                                .build();

                TenantContext context = buildTenantContext(tenantId);

                // Return streaming response
                return orchestrator.streamExecute(
                                inferenceRequest.getModel(),
                                inferenceRequest,
                                context)
                                .map(chunk -> new StreamChunkDTO(
                                                chunk.getIndex(),
                                                chunk.getContent(),
                                                chunk.isFinal()));
        }

        /**
         * Get routing decision info
         */
        @GET
        @Path("/routing/{requestId}")
        @Operation(summary = "Get routing decision", description = "Get routing decision for a request")
        public Response getRoutingDecision(
                        @PathParam("requestId") String requestId) {
                return modelRouter.getLastDecision(requestId)
                                .map(decision -> Response.ok(RoutingDecisionDTO.from(decision)).build())
                                .orElse(Response.status(Response.Status.NOT_FOUND).build());
        }

        private TenantContext buildTenantContext(String tenantId) {
                String userId = securityContext.getUserPrincipal() != null
                                ? securityContext.getUserPrincipal().getName()
                                : "anonymous";

                return TenantContext.builder()
                                .tenantId(tenantId != null ? tenantId : "default")
                                .userId(userId)
                                .build();
        }

}