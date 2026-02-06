package tech.kayys.golek.provider.anthropic;

import io.smallrye.mutiny.Uni;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.HeaderParam;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

@RegisterRestClient(configKey = "anthropic-api")
@Path("/v1")
public interface AnthropicClient {

    @POST
    @Path("/messages")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<AnthropicResponse> createMessage(
            AnthropicRequest request,
            @HeaderParam("x-api-key") String apiKey,
            @HeaderParam("anthropic-version") String version);

    default Uni<AnthropicResponse> createMessage(
            AnthropicRequest request,
            String apiKey) {
        return createMessage(request, apiKey, "2023-06-01");
    }
}