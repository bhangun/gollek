package tech.kayys.gollek.provider.cerebras;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

/**
 * REST client for Cerebras API (OpenAI-compatible)
 */
@RegisterRestClient(configKey = "cerebras-api")
@Path("/v1")
public interface CerebrasClient {

    /**
     * Chat completions (non-streaming)
     */
    @POST
    @Path("/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<CerebrasResponse> chatCompletions(
            @HeaderParam("Authorization") String authorization,
            CerebrasRequest request);

    /**
     * Chat completions (streaming)
     */
    @POST
    @Path("/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Multi<CerebrasStreamResponse> chatCompletionsStream(
            @HeaderParam("Authorization") String authorization,
            CerebrasRequest request);
}