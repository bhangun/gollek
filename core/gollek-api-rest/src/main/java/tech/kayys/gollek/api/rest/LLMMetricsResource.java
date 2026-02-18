
package tech.kayys.gollek.api.rest;

import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import tech.kayys.gollek.metrics.LLMInferenceMetrics;

@Path("/metrics/llm")
public class LLMMetricsResource {

    @Inject
    LLMInferenceMetrics metrics;

    @GET
    @Path("/summary")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getSummary() {
        var snapshot = metrics.getSnapshot();
        return Response.ok(snapshot).build();
    }

    @GET
    @Path("/summary.txt")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getSummaryText() {
        return Response.ok(metrics.getSnapshot().toSummaryString()).build();
    }

    @GET
    @Path("/detailed")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getDetailed() {
        return Response.ok(metrics).build();
    }
}