package tech.kayys.gollek.server.metrics;

import jakarta.annotation.Priority;
import jakarta.inject.Inject;
import jakarta.ws.rs.Priorities;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.container.ContainerRequestFilter;
import jakarta.ws.rs.ext.Provider;

import org.eclipse.microprofile.metrics.MetricRegistry;
import org.eclipse.microprofile.metrics.Counter;

@Provider
@Priority(Priorities.USER)
public class RequestMetricsFilter implements ContainerRequestFilter {

    @Inject
    MetricRegistry registry;

    private final Counter total;

    @Inject
    public RequestMetricsFilter(MetricRegistry registry) {
        this.registry = registry;
        this.total = registry.counter("gollek.requests.total");
    }

    @Override
    public void filter(ContainerRequestContext requestContext) {
        total.inc();
    }
}
