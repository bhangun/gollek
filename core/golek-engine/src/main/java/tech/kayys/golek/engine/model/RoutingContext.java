package tech.kayys.golek.engine.model;

import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.wayang.tenant.TenantContext;

import java.time.Duration;
import java.util.Optional;

public record RoutingContext(
        InferenceRequest request,
        TenantContext tenantContext,
        Optional<String> preferredProvider,
        Optional<String> deviceHint,
        Duration timeout,
        boolean costSensitive,
        int priority) {
}