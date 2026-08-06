package tech.kayys.gollek.server.api.v1;

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

import tech.kayys.gollek.server.SdkProvider;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.gollek.sdk.model.PullProgress;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.Collectors;

@Path("/v1/models")
public class ModelsResource {

    @Inject
    SdkProvider sdkProvider;

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Response listModels(
            @jakarta.ws.rs.QueryParam("runnableOnly") boolean runnableOnly,
            @jakarta.ws.rs.QueryParam("limit") @jakarta.ws.rs.DefaultValue("50") int limit,
            @jakarta.ws.rs.QueryParam("compat") String compat,
            @jakarta.ws.rs.QueryParam("format") String responseFormat,
            @jakarta.ws.rs.QueryParam("namespace") String namespace) {
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
            if (isOpenAiCompat(compat) || isOpenAiCompat(responseFormat)) {
                return Response.ok(AgenticApiMapper.toOpenAiModelsResponse(models)).build();
            }
            return Response.ok(models).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @GET
    @Path("/{id}")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getModelInfo(@PathParam("id") String id) {
        GollekSdk sdk = sdkProvider.getSdk();
        try {
            Optional<ModelInfo> info = sdk.getModelInfo(id);
            if (info.isPresent()) {
                ModelInfo m = info.get();
                return Response.ok(java.util.Map.of(
                        "modelId", m.getModelId(),
                        "format", m.getFormat(),
                        "description", m.getDescription(),
                        "size", m.getSizeBytes())).build();
            } else {
                return Response.status(Response.Status.NOT_FOUND).build();
            }
        } catch (Exception e) {
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
        try {
            String jobId = jobManager.startPullJob(dto.modelSpec(), dto.revision(), dto.force());
            return Response.accepted(java.util.Map.of("status", "pull_started", "jobId", jobId)).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @POST
    @Path("/pull/stream/{jobId}")
    @Produces(MediaType.SERVER_SENT_EVENTS)
    @RestSseElementType(MediaType.APPLICATION_JSON)
    public Multi<PullProgress> pullModelStream(@PathParam("jobId") String jobId) {
        return jobManager.streamProgress(jobId);
    }

    @DELETE
    @Path("/{id}")
    public Response deleteModel(@PathParam("id") String id) {
        GollekSdk sdk = sdkProvider.getSdk();
        try {
            sdk.deleteModel(id);
            return Response.noContent().build();
        } catch (Exception e) {
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
