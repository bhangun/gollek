package tech.kayys.gollek.api.rest;

import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import tech.kayys.gollek.multimodal.ConversationManager;

import tech.kayys.gollek.spi.embedding.EmbeddingService;
import tech.kayys.gollek.spi.embedding.EmbeddingService.EmbeddingResult;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalResponse;



import java.util.*;

/**
 * Phase 2 REST endpoints: conversations, image generation, TTS, and embeddings.
 */
@Path("/v1/multimodal")
@Produces(MediaType.APPLICATION_JSON)
@Tag(name = "Multimodal Extended API")
public class MultimodalExtendedResource {

    @Inject
    ConversationManager conversationManager;
    @Inject
    jakarta.enterprise.inject.Instance<EmbeddingService> embeddingService;


    // =========================================================================
    // Conversations
    // =========================================================================

    @POST
    @Path("/conversations")
    @Consumes(MediaType.APPLICATION_JSON)
    @Operation(summary = "Create a new conversation session")
    public Response createSession(CreateSessionRequest body) {
        if (body == null || body.model() == null) {
            return Response.status(400).entity(Map.of("error", "'model' is required")).build();
        }
        String sessionId = conversationManager.createSession(
                body.model(), body.userId() != null ? body.userId() : "anonymous");
        return Response.ok(Map.of("sessionId", sessionId, "model", body.model())).build();
    }

    @GET
    @Path("/conversations/{sessionId}")
    @Operation(summary = "Get conversation session metadata and history")
    public Response getSession(@PathParam("sessionId") String sessionId) {
        return conversationManager.getSession(sessionId)
                .map(s -> Response.ok(s).build())
                .orElse(Response.status(404)
                        .entity(Map.of("error", "Session not found: " + sessionId))
                        .build());
    }

    @POST
    @Path("/conversations/{sessionId}/chat")
    @Consumes(MediaType.APPLICATION_JSON)
    @Operation(summary = "Send a message turn in an existing conversation")
    public Uni<MultimodalResponse> chat(@PathParam("sessionId") String sessionId,
            ConversationChatRequest body) {
        if (body == null || body.parts() == null || body.parts().isEmpty()) {
            return Uni.createFrom().item(
                    MultimodalResponse.error(UUID.randomUUID().toString(), "unknown",
                            "CONV-400", "'parts' is required"));
        }
        return conversationManager.chat(sessionId, body.parts(),
                body.modelOverride(), body.parameters());
    }

    @DELETE
    @Path("/conversations/{sessionId}")
    @Operation(summary = "Delete a conversation session")
    public Response deleteSession(@PathParam("sessionId") String sessionId) {
        conversationManager.deleteSession(sessionId);
        return Response.ok(Map.of("deleted", sessionId)).build();
    }

    @DELETE
    @Path("/conversations/{sessionId}/history")
    @Operation(summary = "Clear conversation history without deleting session")
    public Response clearHistory(@PathParam("sessionId") String sessionId) {
        conversationManager.clearHistory(sessionId);
        return Response.ok(Map.of("cleared", sessionId)).build();
    }



    // =========================================================================
    // Embeddings
    // =========================================================================

    @POST
    @Path("/embeddings")
    @Consumes(MediaType.APPLICATION_JSON)
    @Operation(summary = "Generate embeddings for text or image content", description = "Returns float32 vectors. Supports batching via 'inputs' array.")
    public Uni<EmbeddingApiResponse> embed(EmbeddingApiRequest request) {
        if (request == null || request.inputs() == null || request.inputs().isEmpty()) {
            return Uni.createFrom().item(new EmbeddingApiResponse(
                    List.of(), 0, "EMB-400", "'inputs' is required"));
        }

        List<MultimodalContent> contents = request.inputs().stream()
                .map(inp -> MultimodalContent.ofText(inp))
                .toList();

        if (embeddingService.isUnsatisfied()) {
            return Uni.createFrom().item(new EmbeddingApiResponse(
                    List.of(), 0, "EMB-501", "Embedding service not available"));
        }

        return embeddingService.get().embedBatch(contents, request.model())
                .map(results -> new EmbeddingApiResponse(results, results.size(), null, null));
    }

    @POST
    @Path("/embeddings/similarity")
    @Consumes(MediaType.APPLICATION_JSON)
    @Operation(summary = "Compute cosine similarity between two texts")
    public Uni<Response> similarity(SimilarityRequest request) {
        MultimodalContent a = MultimodalContent.ofText(request.textA());
        MultimodalContent b = MultimodalContent.ofText(request.textB());

        if (embeddingService.isUnsatisfied()) {
            return Uni.createFrom().item(Response.status(501)
                    .entity(Map.of("error", "Embedding service not available")).build());
        }

        return embeddingService.get().embedBatch(List.of(a, b), request.model())
                .map(results -> {
                    if (results.size() < 2 || results.get(0).hasError()) {
                        return Response.status(500)
                                .entity(Map.of("error", "Embedding failed")).build();
                    }
                    double sim = results.get(0).cosineSimilarity(results.get(1));
                    return Response.ok(Map.of(
                            "similarity", sim,
                            "interpretation", interpretSimilarity(sim))).build();
                });
    }

    // =========================================================================
    // Helpers and DTOs
    // =========================================================================



    private String interpretSimilarity(double sim) {
        if (sim >= 0.95)
            return "nearly identical";
        if (sim >= 0.85)
            return "very similar";
        if (sim >= 0.70)
            return "similar";
        if (sim >= 0.50)
            return "somewhat related";
        if (sim >= 0.25)
            return "loosely related";
        return "unrelated";
    }

    // DTOs
    public record CreateSessionRequest(String model, String userId,
            Map<String, Object> systemPromptParams) {
    }

    public record ConversationChatRequest(List<MultimodalContent> parts,
            String modelOverride,
            Map<String, Object> parameters) {
    }

    public record EmbeddingApiRequest(List<String> inputs, String model) {
    }

    public record EmbeddingApiResponse(List<EmbeddingResult> embeddings, int count,
            String errorCode, String errorMessage) {
    }

    public record SimilarityRequest(String textA, String textB, String model) {
    }
}
