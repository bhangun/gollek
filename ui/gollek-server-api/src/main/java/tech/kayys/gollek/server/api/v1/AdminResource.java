package tech.kayys.gollek.server.api.v1;

import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.DELETE;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import tech.kayys.gollek.server.security.ApiKeyStore;

@Path("/v1/admin/api-keys")
public class AdminResource {

    @Inject
    ApiKeyStore store;

    public static record KeyDTO(String key) { }

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Response listKeys() {
        return Response.ok(store.listKeys()).build();
    }

    @POST
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response addKey(KeyDTO dto) {
        if (dto == null || dto.key() == null || dto.key().isBlank()) {
            return Response.status(Response.Status.BAD_REQUEST).entity(java.util.Map.of("error","key required")).build();
        }
        boolean added = store.addKey(dto.key());
        return Response.status(added ? Response.Status.CREATED : Response.Status.CONFLICT)
                .entity(java.util.Map.of("key", dto.key())).build();
    }

    @DELETE
    @Path("/{key}")
    public Response deleteKey(@PathParam("key") String key) {
        boolean removed = store.removeKey(key);
        return Response.status(removed ? Response.Status.NO_CONTENT : Response.Status.NOT_FOUND).build();
    }
}
