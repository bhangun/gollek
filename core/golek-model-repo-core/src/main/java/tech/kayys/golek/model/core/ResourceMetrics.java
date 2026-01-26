package tech.kayys.golek.model.core;

/**
 * Resource metrics - MOVED to golek-api
 * 
 * @deprecated Use {@link tech.kayys.golek.api.model.ResourceMetrics} instead
 */
@Deprecated(since = "1.1.0", forRemoval = true)
public record ResourceMetrics(
                long cpuUsagePercent,
                long memoryUsageBytes,
                long gpuUsagePercent,
                long vramUsageBytes,
                int activeRequests) {

        // Delegate to API version
        public tech.kayys.golek.api.model.ResourceMetrics toApi() {
                return new tech.kayys.golek.api.model.ResourceMetrics(
                                cpuUsagePercent, memoryUsageBytes, gpuUsagePercent, vramUsageBytes, activeRequests);
        }
}
