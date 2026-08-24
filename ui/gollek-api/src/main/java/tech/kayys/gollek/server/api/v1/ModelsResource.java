package tech.kayys.gollek.server.api.v1;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.DELETE;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import io.smallrye.mutiny.Multi;
import org.jboss.resteasy.reactive.RestSseElementType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tech.kayys.gollek.server.SdkProvider;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.gollek.sdk.model.PullProgress;

import java.util.List;
import java.util.Optional;

@Path("/v1/models")
public class ModelsResource {

    private static final Logger LOG = LoggerFactory.getLogger(ModelsResource.class);

    @Inject
    SdkProvider sdkProvider;
    
    @Inject
    MeterRegistry meterRegistry;

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Response listModels(
            @jakarta.ws.rs.QueryParam("runnableOnly") boolean runnableOnly,
            @jakarta.ws.rs.QueryParam("limit") @jakarta.ws.rs.DefaultValue("50") int limit,
            @jakarta.ws.rs.QueryParam("compat") String compat,
            @jakarta.ws.rs.QueryParam("format") String responseFormat,
            @jakarta.ws.rs.QueryParam("namespace") String namespace) {
        
        Timer.Sample sample = Timer.start(meterRegistry);
        LOG.info("Received REST listModels request");
        
        GollekSdk sdk = sdkProvider.getSdk();
        try {
            tech.kayys.gollek.sdk.model.ModelListRequest request = tech.kayys.gollek.sdk.model.ModelListRequest.builder()
                    .runnableOnly(runnableOnly)
                    .limit(limit)
                    .namespace(namespace)
                    .dedupe(true)
                    .sort(true)
                    .build();
            
            List<ModelInfo> models = sdk.listModels(request);
            
            sample.stop(meterRegistry.timer("gollek.rest.model.list.duration"));
            meterRegistry.counter("gollek.rest.model.list.count").increment();
            LOG.info("Returning {} models for REST listModels request", models.size());
            
            if (isOpenAiCompat(compat) || isOpenAiCompat(responseFormat)) {
                return Response.ok(AgenticApiMapper.toOpenAiModelsResponse(models)).build();
            }
            return Response.ok(models).build();
        } catch (Exception e) {
            LOG.error("Failed REST listModels request", e);
            meterRegistry.counter("gollek.rest.model.list.error").increment();
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @GET
    @Path("/{id}")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getModelInfo(@PathParam("id") String id) {
        Timer.Sample sample = Timer.start(meterRegistry);
        LOG.info("Received REST getModelInfo request for model: {}", id);
        
        GollekSdk sdk = sdkProvider.getSdk();
        try {
            Optional<ModelInfo> info = sdk.getModelInfo(id);
            sample.stop(meterRegistry.timer("gollek.rest.model.info.duration"));
            meterRegistry.counter("gollek.rest.model.info.count").increment();
            
            if (info.isPresent()) {
                ModelInfo m = info.get();
                LOG.info("Returning info for model: {}", id);
                return Response.ok(java.util.Map.of(
                        "modelId", m.getModelId(),
                        "format", m.getFormat(),
                        "description", m.getDescription(),
                        "size", m.getSizeBytes())).build();
            } else {
                LOG.warn("Model not found for info request: {}", id);
                return Response.status(Response.Status.NOT_FOUND).build();
            }
        } catch (Exception e) {
            LOG.error("Failed REST getModelInfo request for model: {}", id, e);
            meterRegistry.counter("gollek.rest.model.info.error").increment();
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @GET
    @Path("/{id}/capabilities")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getModelCapabilities(@PathParam("id") String id) {
        GollekSdk sdk = sdkProvider.getSdk();
        try {
            Optional<ModelInfo> info = sdk.getModelInfo(id);
            java.util.List<Object> providers = safeProviders(sdk);
            return Response.ok(new java.util.HashMap<>()).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    public static record PullRequestDTO(String modelSpec, String revision, boolean force) { }

    @Inject
    tech.kayys.gollek.server.jobs.BackgroundJobManager jobManager;

    @POST
    @Path("/pull")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response pullModel(PullRequestDTO dto) {
        LOG.info("Received REST pullModel request for modelSpec: {}", dto.modelSpec());
        try {
            String jobId = jobManager.startPullJob(dto.modelSpec(), dto.revision(), dto.force());
            meterRegistry.counter("gollek.rest.model.pull.count").increment();
            LOG.info("Started pull job {} for modelSpec: {}", jobId, dto.modelSpec());
            return Response.accepted(java.util.Map.of("status", "pull_started", "jobId", jobId)).build();
        } catch (Exception e) {
            LOG.error("Failed REST pullModel request for modelSpec: {}", dto.modelSpec(), e);
            meterRegistry.counter("gollek.rest.model.pull.error").increment();
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @POST
    @Path("/pull/stream/{jobId}")
    @Produces(MediaType.SERVER_SENT_EVENTS)
    @RestSseElementType(MediaType.APPLICATION_JSON)
    public Multi<PullProgress> pullModelStream(@PathParam("jobId") String jobId) {
        LOG.info("Received REST pull stream request for jobId: {}", jobId);
        return jobManager.streamProgress(jobId)
                .onFailure().invoke(err -> {
                    LOG.error("Error in pull stream for jobId: {}", jobId, err);
                });
    }

    @DELETE
    @Path("/{id}")
    public Response deleteModel(@PathParam("id") String id) {
        LOG.info("Received REST deleteModel request for model: {}", id);
        GollekSdk sdk = sdkProvider.getSdk();
        try {
            sdk.deleteModel(id);
            meterRegistry.counter("gollek.rest.model.delete.count").increment();
            LOG.info("Successfully deleted model: {}", id);
            return Response.noContent().build();
        } catch (Exception e) {
            LOG.error("Failed REST deleteModel request for model: {}", id, e);
            meterRegistry.counter("gollek.rest.model.delete.error").increment();
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    private static boolean isOpenAiCompat(String value) {
        return value != null && "openai".equalsIgnoreCase(value.trim());
    }

    private static java.util.List<Object> safeProviders(GollekSdk sdk) {
        return java.util.List.of();
    }
}
