package tech.kayys.gollek.api.rest;

import jakarta.enterprise.inject.Instance;
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

/**
 * Custom health and diagnostics endpoints
 * Note: Quarkus provides built-in health endpoints at:
 * - /q/health/live (liveness)
 * - /q/health/ready (readiness)
 * - /q/health (combined)
 */
@Path("/health")
@Produces(MediaType.APPLICATION_JSON)
public class HealthResource {

        @Inject
        @Liveness
        Instance<HealthCheck> livenessChecks;

        @Inject
        @Readiness
        Instance<HealthCheck> readinessChecks;

        @GET
        public Response health() {
                return Response.ok(Map.of(
                                "status", "UP",
                                "timestamp", Instant.now(),
                                "info", "Use /q/health/live, /q/health/ready, or /q/health for detailed health checks"))
                                .build();
        }

        @GET
        @Path("/detailed")
        public Response detailedHealth() {
                // Aggregate all liveness checks
                boolean livenessUp = true;
                for (HealthCheck check : livenessChecks) {
                        if (check.call().getStatus() != HealthCheckResponse.Status.UP) {
                                livenessUp = false;
                                break;
                        }
                }

                // Aggregate all readiness checks
                boolean readinessUp = true;
                for (HealthCheck check : readinessChecks) {
                        if (check.call().getStatus() != HealthCheckResponse.Status.UP) {
                                readinessUp = false;
                                break;
                        }
                }

                boolean overall = livenessUp && readinessUp;

                return Response.ok(Map.of(
                                "status", overall ? "UP" : "DOWN",
                                "timestamp", Instant.now(),
                                "liveness", livenessUp ? "UP" : "DOWN",
                                "readiness", readinessUp ? "UP" : "DOWN"))
                                .build();
        }
}