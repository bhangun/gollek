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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.error.ErrorCode;
import tech.kayys.wayang.error.WayangException;
import tech.kayys.wayang.tool.entity.*;
import tech.kayys.wayang.tool.dto.GenerateToolsRequest;
import tech.kayys.wayang.tool.dto.ToolGenerationResult;
import tech.kayys.wayang.tool.dto.InvocationStatus;
import tech.kayys.wayang.tool.service.ToolGenerationService;

import tech.kayys.wayang.tool.dto.ToolExecutionRequest;
import tech.kayys.wayang.tool.dto.ToolExecutionResult;
import tech.kayys.wayang.tool.service.ToolExecutor;

import java.util.*;

/**
 * ============================================================================
 * MCP SERVER REST API
 * ============================================================================
 */
@Path("/mcp")
@Tag(name = "MCP", description = "MCP Server API")
public class McpResource {

        private static final Logger LOG = LoggerFactory.getLogger(McpResource.class);

        @Inject
        ToolGenerationService toolGenerator;

        @Inject
        ToolExecutor toolExecutor;

        /**
         * Generate tools from OpenAPI specification
         */
        @POST
        @Path("/tools/generate")
        @Authenticated
        @Operation(summary = "Generate tools from OpenAPI spec")
        public Uni<RestResponse<ToolGenerationResult>> generateTools(
                        @Valid GenerateToolsRequest request) {

                return toolGenerator.generateTools(request)
                                .map(result -> RestResponse.ok(result))
                                .onFailure().transform(throwable ->
                                                new WayangException(
                                                                ErrorCode.VALIDATION_FAILED,
                                                                "Generation failed: " + throwable.getMessage(),
                                                                throwable));
        }

        /**
         * Execute MCP tool
         */
        @POST
        @Path("/tools/execute")
        @Authenticated
        @Operation(summary = "Execute MCP tool")
        public Uni<RestResponse<ToolExecutionResult>> executeTool(
                        @Valid ToolExecutionRequest request) {

                return toolExecutor.execute(request)
                                .map(result -> RestResponse.ok(result))
                                .onFailure().transform(throwable ->
                                                new WayangException(
                                                                ErrorCode.TOOL_EXECUTION_FAILED,
                                                                "Execution failed: " + throwable.getMessage(),
                                                                throwable));
        }

        /**
         * List available tools
         */
        @GET
        @Path("/tools")
        @Authenticated
        @Operation(summary = "List available tools")
        public Uni<RestResponse<List<McpTool>>> listTools(
                        @QueryParam("namespace") String namespace,
                        @QueryParam("capability") String capability) {

                // This would typically query the database for tools
                // For now, return empty list
                return Uni.createFrom().item(RestResponse.ok(List.of()));
        }

        /**
         * Get tool by ID
         */
        @GET
        @Path("/tools/{toolId}")
        @Authenticated
        @Operation(summary = "Get tool by ID")
        public Uni<RestResponse<McpTool>> getTool(@PathParam("toolId") String toolId) {

                // This would typically query the database for a specific tool
                // For now, return 404
                return Uni.createFrom().item(RestResponse.status(Response.Status.NOT_FOUND));
        }
}
