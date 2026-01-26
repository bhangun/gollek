package tech.kayys.golek.memory;

import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;

import java.util.List;

/**
 * Model management endpoints
 */
@Path("/v1/models")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Models", description = "Model management operations")
public class ModelManagementResource {

        @Inject
        ModelRepository modelRepository;

        @GET
        @Operation(summary = "List models", description = "List all available models")
        public Response listModels(
                        @HeaderParam("X-Tenant-ID") String tenantId,
                        @QueryParam("page") @DefaultValue("0") int page,
                        @QueryParam("size") @DefaultValue("20") int size) {
                // TODO: Implement pagination
                List<ModelManifest> models = modelRepository.findByTenant(
                                new tech.kayys.wayang.inference.api.TenantContext.TenantId(
                                                tenantId != null ? tenantId : "default"),
                                null // Pageable
                );

                List<ModelManifestDTO> dtos = models.stream()
                                .map(ModelManifestDTO::from)
                                .toList();

                return Response.ok(dtos).build();
        }

        @GET
        @Path("/{modelId}")
        @Operation(summary = "Get model", description = "Get model details")
        public Response getModel(
                        @PathParam("modelId") String modelId,
                        @HeaderParam("X-Tenant-ID") String tenantId) {
                return modelRepository.findById(
                                modelId,
                                new tech.kayys.wayang.inference.api.TenantContext.TenantId(
                                                tenantId != null ? tenantId : "default"))
                                .map(ModelManifestDTO::from)
                                .map(dto -> Response.ok(dto).build())
                                .orElse(Response.status(Response.Status.NOT_FOUND).build());
        }

        @POST
        @Operation(summary = "Register model", description = "Register a new model")
        public Response registerModel(
                        ModelManifestDTO request,
                        @HeaderParam("X-Tenant-ID") String tenantId) {
                // TODO: Convert DTO to ModelManifest and save
                return Response.status(Response.Status.CREATED)
                                .entity(request)
                                .build();
        }
}
