package tech.kayys.gollek.server.api.v1;

import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import tech.kayys.gollek.server.SdkProvider;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.provider.ProviderInfo;

import java.util.List;
import java.util.stream.Collectors;

@Path("/v1/providers")
public class ProvidersResource {

    @Inject
    SdkProvider sdkProvider;

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Response listProviders() {
        try {
            GollekSdk sdk = sdkProvider.getSdk();
            List<ProviderInfo> providers = sdk.listAvailableProviders();
            var out = providers.stream().map(p -> java.util.Map.of(
                    "id", p.getProviderId(),
                    "name", p.getName(),
                    "description", p.getDescription())).collect(Collectors.toList());
            return Response.ok(out).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @GET
    @Path("/{id}")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getProviderInfo(@PathParam("id") String id) {
        try {
            GollekSdk sdk = sdkProvider.getSdk();
            ProviderInfo p = sdk.getProviderInfo(id);
            return Response.ok(p).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }

    @POST
    @Path("/{id}/select")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response setPreferred(@PathParam("id") String id) {
        try {
            GollekSdk sdk = sdkProvider.getSdk();
            sdk.setPreferredProvider(id);
            return Response.ok(java.util.Map.of("preferred", id)).build();
        } catch (Exception e) {
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(java.util.Map.of("error", e.getMessage())).build();
        }
    }
}
