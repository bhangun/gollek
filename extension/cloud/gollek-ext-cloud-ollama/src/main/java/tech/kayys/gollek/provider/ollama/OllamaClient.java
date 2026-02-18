package tech.kayys.gollek.provider.ollama;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

/**
 * REST client for Ollama API
 */
@RegisterRestClient(configKey = "ollama-api")
@Path("/api")
public interface OllamaClient {

    /**
     * Chat completion
     */
    @POST
    @Path("/chat")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<OllamaResponse> chat(OllamaRequest request);

    /**
     * Streaming chat completion
     */
    @POST
    @Path("/chat")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Multi<OllamaStreamChunk> chatStream(OllamaRequest request);

    /**
     * Generate embeddings
     */
    @POST
    @Path("/embeddings")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<OllamaEmbeddingResponse> embeddings(OllamaEmbeddingRequest request);

    /**
     * List available models
     */
    @GET
    @Path("/tags")
    @Produces(MediaType.APPLICATION_JSON)
    Uni<OllamaModelsResponse> listModels();

    /**
     * Show model info
     */
    @POST
    @Path("/show")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<OllamaModelInfo> showModel(OllamaShowRequest request);

    /**
     * Pull a model
     */
    @POST
    @Path("/pull")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Multi<OllamaPullProgress> pullModel(OllamaPullRequest request);
}