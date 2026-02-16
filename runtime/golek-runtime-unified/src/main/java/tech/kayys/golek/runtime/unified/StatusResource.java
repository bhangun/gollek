package tech.kayys.golek.runtime.unified;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Readiness;
import org.eclipse.microprofile.health.Liveness;

@Path("/status")
@ApplicationScoped
public class StatusResource {

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String getStatus() {
        return "Golek Unified Runtime is running";
    }

    @GET
    @Path("/health")
    @Produces(MediaType.APPLICATION_JSON)
    @Liveness
    public HealthCheckResponse getLiveness() {
        return HealthCheckResponse.up("golek-unified-runtime");
    }

    @GET
    @Path("/ready")
    @Produces(MediaType.APPLICATION_JSON)
    @Readiness
    public HealthCheckResponse getReadiness() {
        // In a real implementation, you would check if all required services are ready
        return HealthCheckResponse.up("golek-unified-runtime");
    }
}