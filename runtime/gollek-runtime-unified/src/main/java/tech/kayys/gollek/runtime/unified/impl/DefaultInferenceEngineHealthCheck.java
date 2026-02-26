package tech.kayys.gollek.runtime.unified.impl;

import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Liveness;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;
import jakarta.inject.Singleton;

/**
 * A default health check implementation that can be injected with @Default qualifier
 */
@Singleton
@Default
public class DefaultInferenceEngineHealthCheck implements HealthCheck {

    @Override
    public HealthCheckResponse call() {
        return HealthCheckResponse.up("inference-engine-default");
    }
}