package tech.kayys.gollek.server.api.v1;

import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.HttpHeaders;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import org.jboss.resteasy.reactive.SseElementType;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.server.SdkProvider;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

@Path("/v1/completions")
public class InferenceResource {

    @Inject
    SdkProvider sdkProvider;

    @POST
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response createCompletion(@Context HttpHeaders headers, InferenceRequest request) {
        GollekSdk sdk = sdkProvider.getSdk();
        try {
            String apiKey = headers.getHeaderString("X-API-Key");
            if (apiKey != null && (request.getApiKey() == null || request.getApiKey().isBlank())) {
                request = request.toBuilder().apiKey(apiKey).build();
            }
            InferenceResponse resp = sdk.createCompletion(request);
            return Response.ok(resp).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @POST
    @Path("/stream")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.SERVER_SENT_EVENTS)
    @SseElementType(MediaType.APPLICATION_JSON)
    public Multi<StreamingInferenceChunk> streamCompletion(@Context HttpHeaders headers, InferenceRequest request) {
        GollekSdk sdk = sdkProvider.getSdk();
        String apiKey = headers.getHeaderString("X-API-Key");
        if (apiKey != null && (request.getApiKey() == null || request.getApiKey().isBlank())) {
            request = request.toBuilder().apiKey(apiKey).build();
        }
        return sdk.streamCompletion(request);
    }
}
