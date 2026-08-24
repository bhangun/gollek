package tech.kayys.gollek.server.api.v1;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.HttpHeaders;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import org.jboss.resteasy.reactive.RestSseElementType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.server.SdkProvider;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

@Path("/v1/completions")
public class InferenceResource {

    private static final Logger LOG = LoggerFactory.getLogger(InferenceResource.class);

    @Inject
    SdkProvider sdkProvider;
    
    @Inject
    MeterRegistry meterRegistry;

    @POST
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response createCompletion(@Context HttpHeaders headers, InferenceRequest request) {
        Timer.Sample sample = Timer.start(meterRegistry);
        LOG.info("Received REST completion request for model: {}", request.getModel());
        
        GollekSdk sdk = sdkProvider.getSdk();
        try {
            String apiKey = headers.getHeaderString("X-API-Key");
            if (apiKey != null && (request.getApiKey() == null || request.getApiKey().isBlank())) {
                request = request.toBuilder().apiKey(apiKey).build();
            }
            InferenceResponse resp = sdk.createCompletion(request);
            
            sample.stop(meterRegistry.timer("gollek.rest.complete.duration"));
            meterRegistry.counter("gollek.rest.complete.count").increment();
            if (resp.getInputTokens() > 0 || resp.getOutputTokens() > 0) {
                meterRegistry.counter("gollek.rest.complete.tokens").increment(resp.getInputTokens() + resp.getOutputTokens());
            }
            LOG.info("REST completion completed for model: {}, finish reason: {}", request.getModel(), resp.getFinishReason());
            
            return Response.ok(resp).build();
        } catch (Exception e) {
            LOG.error("Failed REST completion request for model: {}", request.getModel(), e);
            meterRegistry.counter("gollek.rest.complete.error").increment();
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @POST
    @Path("/stream")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.SERVER_SENT_EVENTS)
    @RestSseElementType(MediaType.APPLICATION_JSON)
    public Multi<StreamingInferenceChunk> streamCompletion(@Context HttpHeaders headers, InferenceRequest request) {
        LOG.info("Received REST completion stream request for model: {}", request.getModel());
        
        GollekSdk sdk = sdkProvider.getSdk();
        String apiKey = headers.getHeaderString("X-API-Key");
        final InferenceRequest finalRequest = (apiKey != null && (request.getApiKey() == null || request.getApiKey().isBlank()))
                ? request.toBuilder().apiKey(apiKey).build()
                : request;
        
        return sdk.streamCompletion(finalRequest)
                .onItem().invoke(chunk -> meterRegistry.counter("gollek.rest.complete.stream.chunks").increment())
                .onFailure().invoke(err -> {
                    LOG.error("Failed REST completion stream request for model: {}", finalRequest.getModel(), err);
                    meterRegistry.counter("gollek.rest.complete.stream.error").increment();
                });
    }
}
