package tech.kayys.golek.provider.anthropic;

import io.smallrye.mutiny.Multi;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.HeaderParam;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

@RegisterRestClient(configKey = "anthropic-api-streaming")
@Path("/v1")
public interface AnthropicStreamingClient {

    @POST
    @Path("/messages")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.SERVER_SENT_EVENTS) // This is the correct media type for SSE
    Multi<AnthropicStreamResponse> streamMessage(
            AnthropicRequest request,
            @HeaderParam("x-api-key") String apiKey,
            @HeaderParam("anthropic-version") String version);

    default Multi<AnthropicStreamResponse> streamMessage(
            AnthropicRequest request,
            String apiKey) {
        return streamMessage(request, apiKey, "2023-06-01");
    }
}