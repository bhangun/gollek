package com.kmpchatbot.server.client;

import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.HeaderParam;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

@Path("/messages")
@RegisterRestClient(configKey = "anthropic-api")
public interface AnthropicClient {
    
    @POST
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    AnthropicDTO.AnthropicResponse sendMessage(
            @HeaderParam("x-api-key") String apiKey,
            @HeaderParam("anthropic-version") String version,
            AnthropicDTO.AnthropicRequest request
    );
}
