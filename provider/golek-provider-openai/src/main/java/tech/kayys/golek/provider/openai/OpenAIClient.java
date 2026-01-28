package tech.kayys.golek.provider.openai;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

/**
 * REST client for OpenAI API
 */
@RegisterRestClient(configKey = "openai-api")
@Path("/v1")
public interface OpenAIClient {

    /**
     * Chat completions (non-streaming)
     */
    @POST
    @Path("/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<OpenAIResponse> chatCompletions(
            @HeaderParam("Authorization") String authorization,
            OpenAIRequest request);

    /**
     * Chat completions (streaming)
     */
    @POST
    @Path("/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Multi<OpenAIStreamResponse> chatCompletionsStream(
            @HeaderParam("Authorization") String authorization,
            OpenAIRequest request);

    /**
     * Generate embeddings
     */
    @POST
    @Path("/embeddings")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<OpenAIEmbeddingResponse> embeddings(
            @HeaderParam("Authorization") String authorization,
            OpenAIEmbeddingRequest request);

    /**
     * List available models
     */
    @GET
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    Uni<OpenAIModelsResponse> listModels(
            @HeaderParam("Authorization") String authorization);
}