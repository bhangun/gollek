package tech.kayys.golek.memory;

import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Liveness;
import org.eclipse.microprofile.health.Readiness;
import tech.kayys.wayang.inference.kernel.provider.ProviderRegistry;

/**
 * Health check endpoints
 */
@Path("/health")
public class HealthResource {

    @Inject
    ProviderRegistry providerRegistry;

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Response health() {
        return Response.ok()
                .entity(Map.of("status", "UP"))
                .build();
    }

    @Liveness
    public HealthCheckResponse liveness() {
        return HealthCheckResponse.named("inference-server-liveness")
                .up()
                .build();
    }

    @Readiness
    public HealthCheckResponse readiness() {
        boolean allHealthy = providerRegistry.all().stream()
                .allMatch(p -> p.health().isHealthy());

        return HealthCheckResponse.named("inference-server-readiness")
                .status(allHealthy)
                .build();
    }
}