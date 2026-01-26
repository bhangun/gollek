package tech.kayys.golek.runtime.resource;

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

import java.time.Instant;
import java.util.Map;
import tech.kayys.golek.runtime.health.InferenceEngineHealthCheck;

/**
 * Custom health and diagnostics endpoints
 */
@Path("/health")
@Produces(MediaType.APPLICATION_JSON)
public class HealthResource {

        @Inject
        @Liveness
        HealthCheck livenessCheck;

        @Inject
        @Readiness
        HealthCheck readinessCheck;

        @Inject
        InferenceEngineHealthCheck engineHealth;

        @GET
        @Path("/detailed")
        public Response detailedHealth() {
                HealthCheckResponse liveness = livenessCheck.call();
                HealthCheckResponse readiness = readinessCheck.call();
                HealthCheckResponse engine = engineHealth.call();

                boolean overall = liveness.getStatus() == HealthCheckResponse.Status.UP &&
                                readiness.getStatus() == HealthCheckResponse.Status.UP &&
                                engine.getStatus() == HealthCheckResponse.Status.UP;

                return Response.ok(Map.of(
                                "status", overall ? "UP" : "DOWN",
                                "timestamp", Instant.now(),
                                "checks", Map.of(
                                                "liveness", liveness,
                                                "readiness", readiness,
                                                "engine", engine)))
                                .build();
        }
}