package tech.kayys.golek.api.rest;

import io.smallrye.mutiny.Uni;
import jakarta.annotation.security.RolesAllowed;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.ws.rs.*;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.SecurityContext;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.logging.Logger;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.engine.model.ModelManagementService;
import tech.kayys.wayang.tenant.TenantContextResolver;

import java.util.Map;

/**
 * Model management endpoints for administrative operations
 */
@Path("/v1/models")
@Tag(name = "Model Management", description = "Model lifecycle and configuration")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@RolesAllowed({ "admin", "model-manager" })
public class ModelManagementResource {

        private static final Logger LOG = Logger.getLogger(ModelManagementResource.class);

        @Inject
        ModelManagementService modelService;

        @Inject
        TenantContextResolver tenantResolver;

        @GET
        @Operation(summary = "List all models")
        public Uni<Response> listModels(
                        @QueryParam("page") @DefaultValue("0") int page,
                        @QueryParam("size") @DefaultValue("20") int size,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);

                return modelService.listModels(tenantContext, page, size)
                                .map(models -> Response.ok(models).build());
        }

        @GET
        @Path("/{modelId}")
        @Operation(summary = "Get model details")
        public Uni<Response> getModel(
                        @PathParam("modelId") String modelId,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);

                return modelService.getModel(modelId, tenantContext)
                                .map(model -> Response.ok(model).build())
                                .onFailure().recoverWithItem(
                                                Response.status(Response.Status.NOT_FOUND).build());
        }

        @POST
        @Operation(summary = "Register new model")
        public Uni<Response> registerModel(
                        @Valid ModelManifest manifest,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                LOG.infof("Registering model: %s", manifest.modelId());

                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);

                return modelService.registerModel(manifest, tenantContext)
                                .map(registered -> Response
                                                .status(Response.Status.CREATED)
                                                .entity(registered)
                                                .build());
        }

        @PUT
        @Path("/{modelId}")
        @Operation(summary = "Update model")
        public Uni<Response> updateModel(
                        @PathParam("modelId") String modelId,
                        @Valid ModelManifest manifest,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                LOG.infof("Updating model: %s", modelId);

                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);

                return modelService.updateModel(modelId, manifest, tenantContext)
                                .map(updated -> Response.ok(updated).build());
        }

        @DELETE
        @Path("/{modelId}")
        @Operation(summary = "Delete model")
        public Uni<Response> deleteModel(
                        @PathParam("modelId") String modelId,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                LOG.infof("Deleting model: %s", modelId);

                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);

                return modelService.deleteModel(modelId, tenantContext)
                                .map(deleted -> Response.noContent().build());
        }

        @POST
        @Path("/{modelId}/warmup")
        @Operation(summary = "Warmup model")
        public Uni<Response> warmupModel(
                        @PathParam("modelId") String modelId,
                        @Context SecurityContext securityContext,
                        @Context ContainerRequestContext requestContext) {
                LOG.infof("Warming up model: %s", modelId);

                TenantContext tenantContext = resolveTenantContext(securityContext, requestContext);

                return modelService.warmup(modelId, tenantContext)
                                .map(result -> Response
                                                .accepted()
                                                .entity(Map.of("status", "WARMING_UP"))
                                                .build());
        }

        private TenantContext resolveTenantContext(SecurityContext securityContext,
                        ContainerRequestContext requestContext) {
                Object ctx = requestContext != null ? requestContext.getProperty("tenantContext") : null;
                if (ctx instanceof TenantContext tenantContext) {
                        return tenantContext;
                }
                return tenantResolver.resolve(securityContext);
        }
}
