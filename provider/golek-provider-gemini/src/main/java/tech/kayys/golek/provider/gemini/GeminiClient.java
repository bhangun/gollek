package tech.kayys.golek.provider.gemini;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

/**
 * REST client for Google Gemini API
 */
@RegisterRestClient(configKey = "gemini-api")
@Path("/v1beta")
public interface GeminiClient {

    /**
     * Generate content (non-streaming)
     */
    @POST
    @Path("/models/{model}:generateContent")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<GeminiResponse> generateContent(
            @PathParam("model") String model,
            @QueryParam("key") String apiKey,
            GeminiRequest request);

    /**
     * Generate content (streaming)
     */
    @POST
    @Path("/models/{model}:streamGenerateContent")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Multi<GeminiResponse> streamGenerateContent(
            @PathParam("model") String model,
            @QueryParam("key") String apiKey,
            GeminiRequest request);

    /**
     * Generate embeddings
     */
    @POST
    @Path("/models/{model}:embedContent")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<GeminiEmbeddingResponse> embedContent(
            @PathParam("model") String model,
            @QueryParam("key") String apiKey,
            GeminiEmbeddingRequest request);

    /**
     * List available models
     */
    @GET
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    Uni<GeminiModelsResponse> listModels(@QueryParam("key") String apiKey);

    /**
     * Get model info
     */
    @GET
    @Path("/models/{model}")
    @Produces(MediaType.APPLICATION_JSON)
    Uni<GeminiModelInfo> getModel(
            @PathParam("model") String model,
            @QueryParam("key") String apiKey);

    /**
     * Count tokens
     */
    @POST
    @Path("/models/{model}:countTokens")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<GeminiTokenCountResponse> countTokens(
            @PathParam("model") String model,
            @QueryParam("key") String apiKey,
            GeminiRequest request);
}