package tech.kayys.gollek.provider.mistral;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

@Path("/v1")
@RegisterRestClient(baseUri = "https://api.mistral.ai")
public interface MistralClient {

    @POST
    @Path("/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<MistralResponse> chatCompletions(
            @HeaderParam("Authorization") String authorization,
            MistralRequest request);

    @POST
    @Path("/chat/completions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.SERVER_SENT_EVENTS)
    Multi<MistralStreamResponse> chatCompletionsStream(
            @HeaderParam("Authorization") String authorization,
            MistralRequest request);
}
