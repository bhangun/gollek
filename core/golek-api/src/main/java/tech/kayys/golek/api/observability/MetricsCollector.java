package tech.kayys.golek.api.observability;

import tech.kayys.golek.api.tenant.TenantId;
import java.time.Duration;

/**
 * Interface for collecting metrics from inference operations.
 */
public interface MetricsCollector {

    void recordSuccess(
            String provider,
            String model,
            TenantId tenant,
            Duration duration);

    void recordFailure(
            String provider,
            String model,
            TenantId tenant,
            String errorType);
}
