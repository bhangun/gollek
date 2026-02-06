package tech.kayys.golek.model.core;

/**
 * Resource metrics - MOVED to golek-spi
 * 
 * @deprecated Use {@link tech.kayys.golek.spi.model.ResourceMetrics} instead
 */
@Deprecated(since = "1.1.0", forRemoval = true)
public record ResourceMetrics(
                long cpuUsagePercent,
                long memoryUsageBytes,
                long gpuUsagePercent,
                long vramUsageBytes,
                int activeRequests) {

        // Delegate to API version
        public tech.kayys.golek.spi.model.ResourceMetrics toApi() {
                return new tech.kayys.golek.spi.model.ResourceMetrics(
                                cpuUsagePercent, memoryUsageBytes, gpuUsagePercent, vramUsageBytes, activeRequests);
        }
}
