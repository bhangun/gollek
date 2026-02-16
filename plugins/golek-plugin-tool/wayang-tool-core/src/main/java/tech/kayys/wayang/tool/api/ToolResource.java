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

import tech.kayys.wayang.tool.registry.SpecFormatRegistry;
import tech.kayys.wayang.tool.entity.*;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.SourceType;
import tech.kayys.wayang.tool.dto.OpenApiToolRequest;
import tech.kayys.wayang.tool.dto.HttpRequestContext;
import tech.kayys.wayang.tool.dto.RequestContext;
import tech.kayys.wayang.tool.dto.ToolDetailResponse; // Wait, check if this exists
import tech.kayys.wayang.tool.dto.ToolExecuteRequest; // Wait, check if this exists
import tech.kayys.wayang.tool.dto.ToolExecutionResponse; // Wait, check if this exists
import tech.kayys.wayang.tool.dto.ToolGenerationResponse; // Wait, check if this exists
import tech.kayys.wayang.tool.dto.ToolMetadataResponse; // Wait, check if this exists
import tech.kayys.wayang.tool.dto.ToolUpdateRequest; // Wait, check if this exists
import tech.kayys.wayang.tool.dto.InvocationStatus;

import tech.kayys.wayang.tool.repository.ToolRepository;
import tech.kayys.wayang.tool.dto.ToolExecutionRequest;
import tech.kayys.wayang.tool.service.ToolExecutor;

import java.util.*;

/**
 * ============================================================================
 * MCP SERVER REST API
 * ============================================================================
 *
 * RESTful API for MCP tool management and execution.
 *
 * Endpoints:
 * - POST /api/v1/mcp/tools/openapi - Generate tools from OpenAPI
 * - GET /api/v1/mcp/tools - List tools
 * - GET /api/v1/mcp/tools/{toolId} - Get tool details
 * - POST /api/v1/mcp/tools/{toolId}/execute - Execute tool
 * - POST /api/v1/mcp/auth-profiles - Create auth profile
 */

// ==================== TOOL GENERATION API ====================

@Path("/api/v1/mcp/tools")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Authenticated
@Tag(name = "MCP Tools", description = "MCP tool management")
public class ToolResource {

    @Inject
    tech.kayys.wayang.tool.service.ToolGenerationService toolGenerator;

    @Inject
    ToolExecutor toolExecutor;

    @Inject
    RequestContext requestContext;

    @Inject
    SpecFormatRegistry specFormatRegistry;

    @Inject
    ToolRepository mcpToolRepository;

    /**
     * Get supported specification formats
     */
    @GET
    @Path("/formats")
    @Operation(summary = "List supported API specification formats")
    public Map<String, SpecFormatRegistry.SpecFormatInfo> getSupportedFormats() {
        return specFormatRegistry.getSupportedFormats();
    }

    /**
     * Generate tools from OpenAPI specification
     */
    @POST
    @Path("/openapi")
    @Operation(summary = "Generate MCP tools from OpenAPI spec")
    public Uni<RestResponse<ToolGenerationResponse>> generateFromOpenApi(
            @Valid OpenApiToolRequest request) {

        String requestId = requestContext.getCurrentRequestId();

        GenerateToolsRequest genRequest = new GenerateToolsRequest(
                requestId,
                request.namespace(),
                SourceType.valueOf(request.sourceType()),
                request.source(),
                request.authProfileId(),
                "current-user", // From security context
                request.guardrailsConfig() != null ? request.guardrailsConfig() : Map.of());

        return toolGenerator.generateTools(genRequest)
                .map(result -> RestResponse.ok(
                        new ToolGenerationResponse(
                                result.sourceId().toString(),
                                result.namespace(),
                                result.toolsGenerated(),
                                result.toolIds(),
                                result.warnings())))
                .onFailure()
                .transform(error -> new WayangException(ErrorCode.VALIDATION_FAILED, error.getMessage(), error));
    }

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

        String requestId = requestContext.getCurrentRequestId();

        String query = "requestId = ?1";
        List<Object> params = new ArrayList<>();
        params.add(requestId);

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

        String requestId = requestContext.getCurrentRequestId();

        return mcpToolRepository.findByRequestIdAndToolId(requestId, toolId)
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

        String requestId = requestContext.getCurrentRequestId();

        ToolExecutionRequest execRequest = new ToolExecutionRequest(
                requestId,
                "current-user", // userId
                toolId,
                request.arguments(),
                null, // workflow run ID
                null, // agent ID
                request.context() != null ? request.context() : Map.of());

        return toolExecutor.execute(execRequest)
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
                        String errorCode = null;
                        Boolean retryable = null;
                        if (result.metadata() != null) {
                            Object code = result.metadata().get("errorCode");
                            if (code != null) {
                                errorCode = code.toString();
                            }
                            Object retry = result.metadata().get("retryable");
                            if (retry instanceof Boolean) {
                                retryable = (Boolean) retry;
                            }
                        }
                        return RestResponse.status(
                                RestResponse.Status.BAD_REQUEST,
                                new ToolExecutionResponse(
                                        "failure",
                                        Map.of(),
                                        result.errorMessage(),
                                        errorCode,
                                        retryable,
                                        result.executionTimeMs()));
                    }
                });
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

        String requestId = requestContext.getCurrentRequestId();

        return mcpToolRepository.findByRequestIdAndToolId(requestId, toolId)
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

        String requestId = requestContext.getCurrentRequestId();

        return mcpToolRepository.findByRequestIdAndToolId(requestId, toolId)
                .flatMap(tool -> {
                    if (tool == null) {
                        return Uni.createFrom().item(RestResponse.notFound());
                    }

                    return mcpToolRepository.deleteById(toolId)
                            .map(deleted -> deleted ? RestResponse.ok() : RestResponse.notFound());
                });
    }
}
