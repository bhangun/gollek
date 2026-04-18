package tech.kayys.gollek.server.api.v1;

import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import tech.kayys.gollek.server.SdkProvider;
import tech.kayys.gollek.spi.inference.AsyncJobStatus;
import tech.kayys.gollek.spi.inference.InferenceRequest;

@Path("/v1/jobs")
public class JobsResource {

    @Inject
    SdkProvider sdkProvider;

    @POST
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response submitJob(InferenceRequest request) {
        try {
            var sdk = sdkProvider.getSdk();
            String jobId = sdk.submitAsyncJob(request);
            return Response.accepted(java.util.Map.of("jobId", jobId)).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Response listJobs() {
        try {
            var mgr = jakarta.enterprise.inject.spi.CDI.current().select(tech.kayys.gollek.server.jobs.BackgroundJobManager.class).get();
            var list = mgr.listJobs();
            var out = list.stream().map(x -> java.util.Map.of(
                    "jobId", x.jobId(),
                    "status", x.status(),
                    "lastProgress", x.lastProgress(),
                    "error", x.error())).toList();
            return Response.ok(out).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @GET
    @Path("/{id}")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getJobStatus(@PathParam("id") String id) {
        try {
            var opt = jakarta.enterprise.inject.spi.CDI.current()
                    .select(tech.kayys.gollek.server.jobs.BackgroundJobManager.class).get().getJobInfo(id);
            if (opt.isPresent()) {
                var x = opt.get();
                return Response.ok(java.util.Map.of(
                        "jobId", x.jobId(),
                        "status", x.status(),
                        "lastProgress", x.lastProgress(),
                        "error", x.error())).build();
            }
            var sdk = sdkProvider.getSdk();
            AsyncJobStatus status = sdk.getJobStatus(id);
            return Response.ok(status).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @POST
    @Path("/{id}/cancel")
    @Produces(MediaType.APPLICATION_JSON)
    public Response cancelJob(@PathParam("id") String id) {
        try {
            var mgr = jakarta.enterprise.inject.spi.CDI.current().select(tech.kayys.gollek.server.jobs.BackgroundJobManager.class).get();
            boolean cancelled = mgr.cancelJob(id);
            return Response.ok(java.util.Map.of("cancelled", cancelled)).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    // helper to query job manager
    private Object jobStatus(String id) {
        var mgr = jakarta.enterprise.inject.spi.CDI.current().select(tech.kayys.gollek.server.jobs.BackgroundJobManager.class).get();
        var opt = mgr.getJobInfo(id);
        return opt.map(x -> java.util.Map.of("jobId", x.jobId(), "status", x.status(), "lastProgress", x.lastProgress(), "error", x.error())).orElse(null);
    }
}
