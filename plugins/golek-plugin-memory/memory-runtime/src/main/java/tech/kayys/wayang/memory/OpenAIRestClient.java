package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

/**
 * OpenAI REST client interface
 */
@RegisterRestClient(configKey = "openai")
@jakarta.ws.rs.Path("/v1")
public interface OpenAIRestClient {

    @jakarta.ws.rs.POST
    @jakarta.ws.rs.Path("/embeddings")
    @jakarta.ws.rs.Produces(jakarta.ws.rs.core.MediaType.APPLICATION_JSON)
    @jakarta.ws.rs.Consumes(jakarta.ws.rs.core.MediaType.APPLICATION_JSON)
    Uni<OpenAIEmbeddingResponse> createEmbedding(
            @jakarta.ws.rs.HeaderParam("Authorization") String authorization,
            OpenAIEmbeddingRequest request);

    @jakarta.ws.rs.POST
    @jakarta.ws.rs.Path("/embeddings")
    @jakarta.ws.rs.Produces(jakarta.ws.rs.core.MediaType.APPLICATION_JSON)
    @jakarta.ws.rs.Consumes(jakarta.ws.rs.core.MediaType.APPLICATION_JSON)
    Uni<OpenAIEmbeddingResponse> createEmbeddingBatch(
            @jakarta.ws.rs.HeaderParam("Authorization") String authorization,
            OpenAIEmbeddingBatchRequest request);
}