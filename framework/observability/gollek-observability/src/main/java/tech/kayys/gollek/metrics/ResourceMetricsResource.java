package tech.kayys.gollek.metrics;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import java.util.Map;

/**
 * REST endpoint exposing system and JVM resource metrics.
 * Designed to be polled by UI dashboards (e.g. Flutter).
 */
@ApplicationScoped
@Path("/api/v1/metrics/resource")
@Produces(MediaType.APPLICATION_JSON)
public class ResourceMetricsResource {

    @Inject
    ResourceMetrics metrics;

    @GET
    public Map<String, Object> getMetrics() {
        return Map.of(
            "processCpuLoad", metrics.processCpuLoad(),
            "systemCpuLoad", metrics.systemCpuLoad(),
            "heapUsedBytes", metrics.heapUsedBytes(),
            "heapMaxBytes", metrics.heapMaxBytes(),
            "heapUtilization", metrics.heapUtilization(),
            "nonHeapUsedBytes", metrics.nonHeapUsedBytes()
        );
    }
}
