package tech.kayys.wayang.memory.health;

import io.quarkus.hibernate.reactive.panache.Panache;
import io.smallrye.health.checks.UrlHealthCheck;
import io.vertx.mutiny.redis.client.RedisAPI;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Liveness;
import org.eclipse.microprofile.health.Readiness;

import java.util.List;

@ApplicationScoped
public class MemoryServiceHealthCheck {

    @Inject
    RedisAPI redisAPI;

    @Liveness
    public HealthCheck livenessCheck() {
        return () -> HealthCheckResponse.named("memory-service-liveness")
            .up()
            .build();
    }

    @Readiness
    public HealthCheck readinessCheck() {
        return () -> {
            try {
                // Check database connectivity
                Panache.withTransaction(() -> 
                    Panache.getSession()
                        .onItem().transform(session -> true))
                    .await().indefinitely();
                
                // Check Redis connectivity
                redisAPI.ping(List.of("PONG"))
                    .await().indefinitely();
                
                return HealthCheckResponse.named("memory-service-readiness")
                    .up()
                    .build();
            } catch (Exception e) {
                return HealthCheckResponse.named("memory-service-readiness")
                    .down()
                    .withData("error", e.getMessage())
                    .build();
            }
        };
    }

    @Readiness
    public HealthCheck databaseHealthCheck() {
        return () -> {
            try {
                Panache.withTransaction(() -> 
                    Panache.getSession()
                        .onItem().transform(session -> true))
                    .await().indefinitely();
                
                return HealthCheckResponse.named("database")
                    .up()
                    .build();
            } catch (Exception e) {
                return HealthCheckResponse.named("database")
                    .down()
                    .withData("error", e.getMessage())
                    .build();
            }
        };
    }

    @Readiness
    public HealthCheck redisHealthCheck() {
        return () -> {
            try {
                redisAPI.ping(List.of("PONG"))
                    .await().indefinitely();
                
                return HealthCheckResponse.named("redis")
                    .up()
                    .build();
            } catch (Exception e) {
                return HealthCheckResponse.named("redis")
                    .down()
                    .withData("error", e.getMessage())
                    .build();
            }
        };
    }
}