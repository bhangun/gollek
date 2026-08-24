package tech.kayys.gollek.model.domain.download;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import java.util.Collection;

@ApplicationScoped
@Path("/api/v1/downloads")
@Produces(MediaType.APPLICATION_JSON)
public class ModelDownloadResource {

    @Inject
    ModelDownloadManager downloadManager;

    @GET
    public Collection<ModelDownloadState> getActiveDownloads() {
        return downloadManager.getAllActiveDownloads().values();
    }

    @GET
    @Path("/{modelId}")
    public Response getDownloadStatus(@PathParam("modelId") String modelId) {
        ModelDownloadState state = downloadManager.getDownloadState(modelId);
        if (state == null) {
            return Response.status(Response.Status.NOT_FOUND).build();
        }
        return Response.ok(state).build();
    }
}
