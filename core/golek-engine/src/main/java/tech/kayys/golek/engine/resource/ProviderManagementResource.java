package tech.kayys.golek.engine.resource;

import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;

import java.util.List;

/**
 * Provider management endpoints
 */
@Path("/v1/providers")
@Produces(MediaType.APPLICATION_JSON)
@Tag(name = "Providers", description = "Provider management operations")
public class ProviderManagementResource {

    @Inject
    ProviderRegistry providerRegistry;

    @Inject
    CircuitBreakerRegistry circuitBreakerRegistry;

    @GET
    @Operation(summary = "List providers", description = "List all available providers")
    public Response listProviders() {
        List<ProviderInfoDTO> providers = providerRegistry.all().stream()
                .map(provider -> {
                    var health = provider.health();
                    var breaker = circuitBreakerRegistry.get(provider.id());

                    return new ProviderInfoDTO(
                            provider.id(),
                            provider.name(),
                            provider.capabilities(),
                            health.isHealthy(),
                            health.getMessage(),
                            breaker.map(b -> b.getState().toString()).orElse("UNKNOWN"));
                })
                .toList();

        return Response.ok(providers).build();
    }

    @GET
    @Path("/{providerId}")
    @Operation(summary = "Get provider", description = "Get provider details")
    public Response getProvider(@PathParam("providerId") String providerId) {
        return providerRegistry.get(providerId)
                .map(provider -> {
                    var health = provider.health();
                    var breaker = circuitBreakerRegistry.get(provider.id());

                    ProviderInfoDTO dto = new ProviderInfoDTO(
                            provider.id(),
                            provider.name(),
                            provider.capabilities(),
                            health.isHealthy(),
                            health.getMessage(),
                            breaker.map(b -> b.getState().toString()).orElse("UNKNOWN"));

                    return Response.ok(dto).build();
                })
                .orElse(Response.status(Response.Status.NOT_FOUND).build());
    }

    @POST
    @Path("/{providerId}/circuit-breaker/reset")
    @Operation(summary = "Reset circuit breaker", description = "Reset provider circuit breaker")
    public Response resetCircuitBreaker(@PathParam("providerId") String providerId) {
        circuitBreakerRegistry.reset(providerId);
        return Response.ok().build();
    }
}