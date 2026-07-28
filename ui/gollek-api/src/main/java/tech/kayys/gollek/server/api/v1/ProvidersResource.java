package tech.kayys.gollek.server.api.v1;

import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import tech.kayys.gollek.sdk.core.GollekSdk;

import java.util.List;

@Path("/api/v1/providers")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class ProvidersResource {

    @Inject
    GollekSdk sdk;

    @GET
    public List<Object> listProviders() {
        return List.of();
    }

    @GET
    @Path("/{id}")
    public Response getProvider(@PathParam("id") String id) {
        return Response.status(Response.Status.NOT_FOUND).build();
    }

    @POST
    @Path("/{id}/preferred")
    public Response setPreferredProvider(@PathParam("id") String id) {
        return Response.ok().build();
    }
}
