package tech.kayys.golek.api.rest;

import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import tech.kayys.golek.engine.registry.CircuitBreakerRegistry;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;

import tech.kayys.golek.spi.provider.ProviderRegistry;
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
    public Uni<Response> listProviders() {
        List<Uni<ProviderInfoDTO>> unis = providerRegistry.getAllProviders().stream()
                .map(this::mapToDTO)
                .toList();

        if (unis.isEmpty()) {
            return Uni.createFrom().item(Response.ok(List.of()).build());
        }

        return Uni.join().all(unis).andCollectFailures()

                .map(list -> Response.ok(list).build());
    }

    @GET
    @Path("/{providerId}")
    @Operation(summary = "Get provider", description = "Get provider details")
    public Uni<Response> getProvider(@PathParam("providerId") String providerId) {
        return Uni.createFrom().optional(providerRegistry.getProvider(providerId))
                .onItem().ifNotNull()
                .transformToUni(provider -> mapToDTO(provider).map(dto -> Response.ok(dto).build()))
                .onItem().ifNull().continueWith(Response.status(Response.Status.NOT_FOUND).build());
    }

    private Uni<ProviderInfoDTO> mapToDTO(tech.kayys.golek.spi.provider.LLMProvider provider) {
        var breaker = circuitBreakerRegistry.get(provider.id());
        String breakerState = breaker.map(b -> b.getState().toString()).orElse("UNKNOWN");

        return provider.health()
                .map(health -> new ProviderInfoDTO(
                        provider.id(),
                        provider.name(),
                        provider.capabilities(),
                        health.isHealthy(),
                        health.message(),
                        breakerState))
                // Fallback if health check fails
                .onFailure().recoverWithItem(new ProviderInfoDTO(
                        provider.id(),
                        provider.name(),
                        provider.capabilities(),
                        false,
                        "Health check failed: " + provider.id(),
                        breakerState));
    }

    @POST
    @Path("/{providerId}/circuit-breaker/reset")
    @Operation(summary = "Reset circuit breaker", description = "Reset provider circuit breaker")
    public Response resetCircuitBreaker(@PathParam("providerId") String providerId) {
        circuitBreakerRegistry.reset(providerId);
        return Response.ok().build();
    }
}