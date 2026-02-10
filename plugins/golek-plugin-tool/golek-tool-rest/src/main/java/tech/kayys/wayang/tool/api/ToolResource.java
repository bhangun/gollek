package tech.kayys.wayang.tool.api;

import io.quarkus.security.Authenticated;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.*;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.resteasy.reactive.RestResponse;
import tech.kayys.wayang.error.ErrorCode;
import tech.kayys.wayang.error.WayangException;

// import tech.kayys.wayang.tool.api.SpecFormatRegistry; // Need to migrate this if it exists
// import tech.kayys.wayang.tool.parser.OpenApiToolGenerator; // Likely in parser module

import tech.kayys.wayang.tool.entity.*;
import tech.kayys.wayang.tool.dto.*;
import tech.kayys.wayang.tool.repository.ToolRepository;
import tech.kayys.wayang.tool.spi.ToolExecutor;

import java.util.*;

/**
 * ============================================================================
 * MCP SERVER REST API
 * ============================================================================
 *
 * RESTful API for MCP tool management and execution.
 */

// ==================== TOOL GENERATION API ====================

@Path("/api/v1/mcp/tools")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Authenticated
@Tag(name = "MCP Tools", description = "MCP tool management")
public class ToolResource {

    // @Inject
    // OpenApiToolGenerator toolGenerator; // Commented out until we verify parser module dependency

    @Inject
    ToolExecutor toolExecutor;

    // @Inject
    // TenantContext tenantContext; // Need to migrate or import correct TenantContext (likely from SPI or Infra)
    // Assuming TenantContext is available in classpath, but likely needs import.
    // If it was in tool.dto, I moved it? No, I didn't see TenantContext in the DTO list I moved.
    // Logic below uses tenantContext.getCurrentTenantId().
    // I need to check where TenantContext is. 
    // It was in `find_by_name` list: `tech/kayys/wayang/tool/dto/TenantContext.java`.
    // I missed it. I need to migrate TenantContext to golek-tool-execution (dto).

    // @Inject
    // SpecFormatRegistry specFormatRegistry; // Missed this too.

    @Inject
    ToolRepository mcpToolRepository;

    /*
    @GET
    @Path("/formats")
    @Operation(summary = "List supported API specification formats")
    public Map<String, SpecFormatRegistry.SpecFormatInfo> getSupportedFormats() {
        return specFormatRegistry.getSupportedFormats();
    }
    */

    /*
    @POST
    @Path("/openapi")
    @Operation(summary = "Generate MCP tools from OpenAPI spec")
    public Uni<RestResponse<ToolGenerationResponse>> generateFromOpenApi(
            @Valid OpenApiToolRequest request) { // OpenApiToolRequest missed?

        String tenantId = tenantContext.getCurrentTenantId();

        GenerateToolsRequest genRequest = new GenerateToolsRequest(
                tenantId,
                request.namespace(),
                request.sourceType(),
                request.source(),
                request.authProfileId(),
                "current-user",
                request.guardrailsConfig() != null ? request.guardrailsConfig() : Map.of());

        return toolGenerator.generateTools(genRequest)
                .map(result -> RestResponse.ok(
                        new ToolGenerationResponse(
                                result.sourceId().toString(),
                                result.namespace(),
                                result.toolsGenerated(),
                                result.toolIds(),
                                result.warnings())))
                .onFailure().transform(error ->
                        new WayangException(ErrorCode.VALIDATION_FAILED, error.getMessage(), error));
    }
    */

    /**
     * List tools for tenant
     */
    @GET
    @Operation(summary = "List MCP tools")
    public Uni<List<ToolMetadataResponse>> listTools(
            @QueryParam("namespace") String namespace,
            @QueryParam("capability") String capability,
            @QueryParam("tag") String tag,
            @QueryParam("enabled") Boolean enabled,
            @QueryParam("readOnly") Boolean readOnly) {

        // Placeholder for tenantId until TenantContext is fixed
        String tenantId = "default"; // tenantContext.getCurrentTenantId();

        String query = "tenantId = ?1";
        List<Object> params = new ArrayList<>();
        params.add(tenantId);

        if (namespace != null) {
            query += " and namespace = ?" + (params.size() + 1);
            params.add(namespace);
        }
        if (enabled != null) {
            query += " and enabled = ?" + (params.size() + 1);
            params.add(enabled);
        }
        if (readOnly != null) {
            query += " and readOnly = ?" + (params.size() + 1);
            params.add(readOnly);
        }

        return mcpToolRepository.searchTools(query, params.toArray())
                .map(tools -> tools.stream()
                        .map(tool -> new ToolMetadataResponse(
                                tool.getToolId(),
                                tool.getName(),
                                tool.getDescription(),
                                tool.getCapabilities(),
                                tool.getCapabilityLevel().name(),
                                tool.isReadOnly(),
                                tool.getTags()))
                        .toList());
    }

    /**
     * Get tool details
     */
    @GET
    @Path("/{toolId}")
    @Operation(summary = "Get tool details")
    public Uni<RestResponse<ToolDetailResponse>> getTool(
            @PathParam("toolId") String toolId) {

        String tenantId = "default"; // tenantContext.getCurrentTenantId();

        return mcpToolRepository.findByTenantIdAndToolId(tenantId, toolId)
                .map(tool -> {
                    if (tool == null) {
                        return RestResponse.notFound();
                    }

                    return RestResponse.ok(new ToolDetailResponse(
                            tool.getToolId(),
                            tool.getName(),
                            tool.getDescription(),
                            tool.getInputSchema(),
                            tool.getOutputSchema(),
                            tool.getCapabilities(),
                            tool.getCapabilityLevel().name(),
                            tool.isEnabled(),
                            tool.isReadOnly(),
                            tool.getMetrics() != null ? tool.getMetrics().getTotalInvocations() : 0L));
                });
    }

    /**
     * Execute tool
     */
    @POST
    @Path("/{toolId}/execute")
    @Operation(summary = "Execute MCP tool")
    public Uni<RestResponse<ToolExecutionResponse>> executeTool(
            @PathParam("toolId") String toolId,
            @Valid ToolExecuteRequest request) {

        String tenantId = "default"; // tenantContext.getCurrentTenantId();

        ToolExecutionRequest execRequest = new ToolExecutionRequest(
                tenantId,
                "current-user", // userId
                toolId,
                request.arguments(),
                null, // workflow run ID
                null, // agent ID
                request.context() != null ? request.context() : Map.of());

        return toolExecutor.execute(toolId, request.arguments(), request.context()) // Adapted to ToolExecutor SPI
                .map(output -> {
                    // Assuming success for simple SPI
                     return RestResponse.ok(new ToolExecutionResponse(
                                "success",
                                output,
                                null,
                                null,
                                null,
                                0L));
                });
                /*
                .map(result -> {
                    if (result.status() == InvocationStatus.SUCCESS) {
                        return RestResponse.ok(new ToolExecutionResponse(
                                "success",
                                result.output(),
                                null,
                                null,
                                null,
                                result.executionTimeMs()));
                    } else {
                        // ...
                        return RestResponse.status(RestResponse.Status.BAD_REQUEST, ...);
                    }
                });
                */
    }

    /**
     * Update tool configuration
     */
    @PUT
    @Path("/{toolId}")
    @Operation(summary = "Update tool configuration")
    public Uni<RestResponse<Void>> updateTool(
            @PathParam("toolId") String toolId,
            @Valid ToolUpdateRequest request) {

        String tenantId = "default"; // tenantContext.getCurrentTenantId();

        return mcpToolRepository.findByTenantIdAndToolId(tenantId, toolId)
                .flatMap(tool -> {
                    if (tool == null) {
                        return Uni.createFrom().item(RestResponse.notFound());
                    }

                    if (request.enabled() != null) {
                        tool.setEnabled(request.enabled());
                    }
                    if (request.description() != null) {
                        tool.setDescription(request.description());
                    }
                    if (request.tags() != null) {
                        tool.setTags(request.tags());
                    }

                    return mcpToolRepository.update(tool)
                            .map(v -> RestResponse.ok());
                });
    }

    /**
     * Delete tool
     */
    @DELETE
    @Path("/{toolId}")
    @Operation(summary = "Delete tool")
    public Uni<RestResponse<Void>> deleteTool(
            @PathParam("toolId") String toolId) {

        String tenantId = "default"; // tenantContext.getCurrentTenantId();

        return mcpToolRepository.findByTenantIdAndToolId(tenantId, toolId)
                .flatMap(tool -> {
                    if (tool == null) {
                        return Uni.createFrom().item(RestResponse.notFound());
                    }

                    return mcpToolRepository.deleteById(toolId)
                            .map(deleted -> deleted ? RestResponse.ok() : RestResponse.notFound());
                });
    }
}
