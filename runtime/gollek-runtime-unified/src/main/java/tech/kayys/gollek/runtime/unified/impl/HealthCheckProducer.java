package tech.kayys.gollek.runtime.unified.impl;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Singleton;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Liveness;
import tech.kayys.gollek.observability.InferenceEngineHealthCheck;

@ApplicationScoped
public class HealthCheckProducer {
    
    @Produces
    @Default
    public InferenceEngineHealthCheck produceInferenceEngineHealthCheck() {
        // Create a default implementation that satisfies the dependency
        return new InferenceEngineHealthCheck() {
            @Override
            public HealthCheckResponse call() {
                return HealthCheckResponse.up("inference-engine");
            }
        };
    }
}