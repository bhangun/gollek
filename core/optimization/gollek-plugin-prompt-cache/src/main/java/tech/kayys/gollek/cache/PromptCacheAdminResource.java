package tech.kayys.gollek.promptcache;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.DELETE;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.logging.Logger;
import tech.kayys.gollek.cache.PromptCacheStats;
import tech.kayys.gollek.cache.PromptCacheStore;

/**
 * REST admin API for the prompt-cache subsystem.
 *
 * <p>All endpoints are under {@code /admin/prompt-cache} and are intended
 * for operator use only. Secure behind an internal/admin firewall or
 * Quarkus security policy in production.
 *
 * <h3>Endpoints</h3>
 * <pre>
 * GET    /admin/prompt-cache/stats                 — live statistics snapshot
 * DELETE /admin/prompt-cache/all                   — flush entire cache
 * DELETE /admin/prompt-cache/model/{modelId}       — flush entries for a model
 * DELETE /admin/prompt-cache/session/{sessionId}   — flush entries for a session
 * </pre>
 */
@ApplicationScoped
@Path("/admin/prompt-cache")
@Produces(MediaType.APPLICATION_JSON)
@Tag(name = "Prompt Cache Admin")
public class PromptCacheAdminResource {

    private static final Logger LOG = Logger.getLogger(PromptCacheAdminResource.class);

    @Inject PromptCacheStore store;

    @GET
    @Path("/stats")
    @Operation(summary = "Live prompt-cache statistics")
    public PromptCacheStats stats() {
        return store.stats();
    }

    @DELETE
    @Path("/all")
    @Operation(summary = "Flush entire prompt cache")
    public Response flushAll() {
        LOG.warn("[PromptCacheAdmin] full cache flush requested");
        store.invalidateAll();
        return Response.ok(java.util.Map.of("flushed", "all")).build();
    }

    @DELETE
    @Path("/model/{modelId}")
    @Operation(summary = "Flush all entries for a specific model")
    public Response flushModel(@PathParam("modelId") String modelId) {
        LOG.infof("[PromptCacheAdmin] flush model=%s", modelId);
        store.invalidateByModel(modelId);
        return Response.ok(java.util.Map.of("flushed", "model:" + modelId)).build();
    }

    @DELETE
    @Path("/session/{sessionId}")
    @Operation(summary = "Flush all entries for a specific session")
    public Response flushSession(@PathParam("sessionId") String sessionId) {
        LOG.infof("[PromptCacheAdmin] flush session=%s", sessionId);
        store.invalidateBySession(sessionId);
        return Response.ok(java.util.Map.of("flushed", "session:" + sessionId)).build();
    }
}
