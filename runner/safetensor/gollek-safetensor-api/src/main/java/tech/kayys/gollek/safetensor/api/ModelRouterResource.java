/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;

import java.nio.file.Files;
import java.util.Map;

/**
 * REST control plane for A/B model routing.
 * Uses reflection to access ModelRouter to avoid circular dependencies in Maven.
 */
@jakarta.ws.rs.Path("/v1/routing")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Routing", description = "A/B model routing and canary deploys")
public class ModelRouterResource {

    private static final org.jboss.logging.Logger log = org.jboss.logging.Logger.getLogger(ModelRouterResource.class);

    @Inject
    jakarta.enterprise.inject.Instance<Object> routerInstance;

    @ConfigProperty(name = "gollek.admin.api-key", defaultValue = "")
    String adminKey;

    private Object getRouter() {
        try {
            return routerInstance.get();
        } catch (Exception e) {
            log.error("ModelRouter bean not found in CDI context", e);
            return null;
        }
    }

    @GET
    @jakarta.ws.rs.Path("/stats")
    @Operation(summary = "Current routing statistics")
    public Response stats(@HeaderParam("X-Admin-Key") String key) {
        checkAuth(key);
        Object router = getRouter();
        if (router == null) return Response.status(503).entity(Map.of("error", "Router not available")).build();
        
        try {
            var stats = router.getClass().getMethod("stats").invoke(router);
            return Response.ok(stats).build();
        } catch (Exception e) {
            return Response.serverError().entity(Map.of("error", e.getMessage())).build();
        }
    }

    @PUT
    @jakarta.ws.rs.Path("/mode")
    @Operation(summary = "Set routing mode (passthrough | shadow | canary | a_b)")
    public Response setMode(@HeaderParam("X-Admin-Key") String key, ModeRequest req) {
        checkAuth(key);
        Object router = getRouter();
        if (router == null) return Response.status(503).entity(Map.of("error", "Router not available")).build();

        try {
            Class<?> modeEnum = Class.forName("tech.kayys.gollek.safetensor.engine.generation.ModelRouter$Mode");
            Object mode = Enum.valueOf((Class<Enum>) modeEnum, req.mode.toUpperCase().replace("-", "_"));
            router.getClass().getMethod("setMode", modeEnum, int.class).invoke(router, mode, req.canaryPct != null ? req.canaryPct : 5);
            return Response.ok(Map.of("status", "updated", "mode", req.mode, "canaryPct", req.canaryPct)).build();
        } catch (Exception e) {
            return Response.serverError().entity(Map.of("error", e.getMessage())).build();
        }
    }

    @POST
    @jakarta.ws.rs.Path("/models/primary")
    @Operation(summary = "Set the primary model")
    public Response setPrimary(@HeaderParam("X-Admin-Key") String key, ModelRef req) {
        checkAuth(key);
        Object router = getRouter();
        if (router == null) return Response.status(503).entity(Map.of("error", "Router not available")).build();

        java.nio.file.Path p = java.nio.file.Path.of(req.path);
        if (!Files.exists(p))
            return Response.status(404).entity(Map.of("error", "Path not found: " + req.path)).build();

        try {
            router.getClass().getMethod("setPrimary", java.nio.file.Path.class).invoke(router, p);
            return Response.ok(Map.of("status", "primary set", "path", req.path)).build();
        } catch (Exception e) {
            return Response.serverError().entity(Map.of("error", e.getMessage())).build();
        }
    }

    @POST
    @jakarta.ws.rs.Path("/models/secondary")
    @Operation(summary = "Set the secondary (candidate) model")
    public Response setSecondary(@HeaderParam("X-Admin-Key") String key, ModelRef req) {
        checkAuth(key);
        Object router = getRouter();
        if (router == null) return Response.status(503).entity(Map.of("error", "Router not available")).build();

        java.nio.file.Path p = java.nio.file.Path.of(req.path);
        if (!Files.exists(p))
            return Response.status(404).entity(Map.of("error", "Path not found: " + req.path)).build();

        try {
            router.getClass().getMethod("setSecondary", java.nio.file.Path.class).invoke(router, p);
            return Response.ok(Map.of("status", "secondary set", "path", req.path)).build();
        } catch (Exception e) {
            return Response.serverError().entity(Map.of("error", e.getMessage())).build();
        }
    }

    @PUT
    @jakarta.ws.rs.Path("/promote")
    @Operation(summary = "Promote secondary to primary (swap models)")
    public Response promote(@HeaderParam("X-Admin-Key") String key) {
        checkAuth(key);
        Object router = getRouter();
        if (router == null) return Response.status(503).entity(Map.of("error", "Router not available")).build();

        try {
            router.getClass().getMethod("promote").invoke(router);
            return Response.ok(Map.of("status", "promoted")).build();
        } catch (Exception e) {
            return Response.serverError().entity(Map.of("error", e.getMessage())).build();
        }
    }

    private void checkAuth(String provided) {
        if (!adminKey.isBlank() && !adminKey.equals(provided))
            throw new WebApplicationException(Response.status(401)
                    .entity(Map.of("error", "invalid admin key")).build());
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class ModeRequest {
        @JsonProperty("mode")
        public String mode;
        @JsonProperty("canary_pct")
        public Integer canaryPct;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class ModelRef {
        @JsonProperty("path")
        public String path;
        @JsonProperty("alias")
        public String alias;
    }
}
