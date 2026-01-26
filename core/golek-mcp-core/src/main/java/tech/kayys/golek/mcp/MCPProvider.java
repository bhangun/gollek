package tech.kayys.golek.provider.core.mcp;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.provider.core.plugin.LLMProvider;
import tech.kayys.golek.provider.core.provider.ProviderCapabilities;
import tech.kayys.golek.provider.core.provider.ProviderRequest;

import org.jboss.logging.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * MCP-based LLM provider that supports tools, resources, and prompts.
 * Integrates with the inference kernel's provider abstraction.
 */
@ApplicationScoped
public class MCPProvider implements LLMProvider {

    private static final Logger LOG = Logger.getLogger(MCPProvider.class);
    private static final String PROVIDER_ID = "mcp";

    @Inject
    MCPClient mcpClient;

    @Inject
    MCPToolExecutor toolExecutor;

    @Inject
    MCPToolRegistry toolRegistry;

    @Inject
    MCPResourceProvider resourceProvider;

    @Inject
    MCPPromptProvider promptProvider;

    private final Map<String, MCPProviderConfig> configurations = new ConcurrentHashMap<>();
    private volatile boolean initialized = false;

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(false) // MCP typically doesn't support streaming
                .tools(true)
                .multimodal(true) // Via resources
                .maxContextTokens(128000) // Depends on underlying model
                .functionCalling(true)
                .build();
    }

    /**
     * Initialize provider with MCP server connections
     */
    public Uni<Void> initialize(List<MCPClientConfig> serverConfigs) {
        if (initialized) {
            return Uni.createFrom().voidItem();
        }

        LOG.info("Initializing MCP provider with %d server(s)", serverConfigs.size());

        return Uni.combine().all().unis(
                serverConfigs.stream()
                        .map(config -> mcpClient.connect(config)
                                .onItem().invoke(connection -> {
                                    toolRegistry.registerConnection(connection);
                                    resourceProvider.registerConnection(connection);
                                    promptProvider.registerConnection(connection);

                                    configurations.put(config.getName(),
                                            new MCPProviderConfig(config.getName(), config));
                                }))
                        .toList())
                .discardItems()
                .onItem().invoke(() -> {
                    initialized = true;
                    LOG.infof("MCP provider initialized with %d connections", serverConfigs.size());
                });
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        if (!initialized) {
            return Uni.createFrom().failure(
                    new IllegalStateException("MCP provider not initialized"));
        }

        long startTime = System.currentTimeMillis();
        LOG.debugf("Processing MCP inference request: %s", request.getRequestId());

        return processRequest(request)
                .onItem().transform(result -> buildResponse(request, result, startTime))
                .onFailure().recoverWithItem(error -> {
                    LOG.errorf(error, "MCP inference failed for request: %s", request.getRequestId());
                    return buildErrorResponse(request, error, startTime);
                });
    }

    /**
     * Process inference request through MCP workflow
     */
    private Uni<MCPInferenceResult> processRequest(ProviderRequest request) {
        var context = MCPInferenceContext.builder()
                .requestId(request.getRequestId())
                .model(request.getModel())
                .messages(request.getMessages())
                .parameters(request.getParameters())
                .build();

        // Step 1: Check for tool calls in request
        return extractToolCalls(context)
                .onItem().transformToUni(toolCalls -> {
                    if (!toolCalls.isEmpty()) {
                        // Execute tools
                        return executeTools(toolCalls, context)
                                .onItem().transformToUni(toolResults -> processToolResults(context, toolResults));
                    } else {
                        // Check for prompt execution
                        return checkPromptExecution(context)
                                .onItem().transformToUni(promptResult -> {
                                    if (promptResult != null) {
                                        return Uni.createFrom().item(
                                                MCPInferenceResult.fromPrompt(promptResult));
                                    } else {
                                        // Check for resource access
                                        return checkResourceAccess(context)
                                                .onItem().transform(resourceContent -> {
                                                    if (resourceContent != null) {
                                                        return MCPInferenceResult.fromResource(resourceContent);
                                                    } else {
                                                        // No MCP-specific processing, return as-is
                                                        return MCPInferenceResult.fromMessages(context.getMessages());
                                                    }
                                                });
                                    }
                                });
                    }
                });
    }

    /**
     * Extract tool calls from request messages
     */
    private Uni<Map<String, Map<String, Object>>> extractToolCalls(MCPInferenceContext context) {
        return Uni.createFrom().item(() -> {
            Map<String, Map<String, Object>> toolCalls = new HashMap<>();

            // Look for tool calls in parameters
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> tools = (List<Map<String, Object>>) context.getParameters().get("tools");

            if (tools != null && !tools.isEmpty()) {
                tools.forEach(tool -> {
                    String toolName = (String) tool.get("name");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> arguments = (Map<String, Object>) tool.get("arguments");
                    if (toolName != null && arguments != null) {
                        toolCalls.put(toolName, arguments);
                    }
                });
            }

            return toolCalls;
        });
    }

    /**
     * Execute multiple tools in parallel
     */
    private Uni<Map<String, MCPToolResult>> executeTools(
            Map<String, Map<String, Object>> toolCalls,
            MCPInferenceContext context) {
        LOG.debugf("Executing %d MCP tools", toolCalls.size());
        return toolExecutor.executeTools(toolCalls)
                .onItem().invoke(results -> LOG.debugf("Executed %d tools, %d succeeded",
                        results.size(),
                        results.values().stream().filter(MCPToolResult::isSuccess).count()));
    }

    /**
     * Process tool results and build response
     */
    private Uni<MCPInferenceResult> processToolResults(
            MCPInferenceContext context,
            Map<String, MCPToolResult> toolResults) {
        return Uni.createFrom().item(() -> {
            // Aggregate tool results
            StringBuilder resultText = new StringBuilder();
            Map<String, Object> metadata = new HashMap<>();

            toolResults.forEach((toolName, result) -> {
                if (result.isSuccess()) {
                    resultText.append("Tool: ").append(toolName).append("\n");
                    resultText.append(result.getAllText()).append("\n\n");
                } else {
                    resultText.append("Tool ").append(toolName)
                            .append(" failed: ").append(result.getErrorMessage())
                            .append("\n\n");
                }
            });

            metadata.put("toolResults", toolResults);
            metadata.put("toolCount", toolResults.size());

            return new MCPInferenceResult(
                    resultText.toString().trim(),
                    metadata,
                    0 // Token count unknown for MCP
            );
        });
    }

    /**
     * Check if request is for prompt execution
     */
    private Uni<tech.kayys.wayang.inference.providers.mcp.prompts.MCPPromptResult> checkPromptExecution(
            MCPInferenceContext context) {
        return Uni.createFrom().item(() -> {
            // Look for prompt name in parameters
            String promptName = (String) context.getParameters().get("prompt");
            if (promptName != null) {
                @SuppressWarnings("unchecked")
                Map<String, String> promptArgs = (Map<String, String>) context.getParameters().get("prompt_arguments");

                return promptProvider.executePrompt(
                        promptName,
                        promptArgs != null ? promptArgs : Map.of()).await().indefinitely();
            }
            return null;
        });
    }

    /**
     * Check if request needs resource access
     */
    private Uni<String> checkResourceAccess(MCPInferenceContext context) {
        return Uni.createFrom().item(() -> {
            // Look for resource URIs in parameters
            @SuppressWarnings("unchecked")
            List<String> resourceUris = (List<String>) context.getParameters().get("resources");

            if (resourceUris != null && !resourceUris.isEmpty()) {
                return resourceProvider.readResources(resourceUris)
                        .await().indefinitely()
                        .values().stream()
                        .map(content -> content.getContentAsString())
                        .reduce("", (a, b) -> a + "\n\n" + b)
                        .trim();
            }
            return null;
        });
    }

    /**
     * Build successful inference response
     */
    private InferenceResponse buildResponse(
            ProviderRequest request,
            MCPInferenceResult result,
            long startTime) {
        long duration = System.currentTimeMillis() - startTime;

        return InferenceResponse.builder()
                .requestId(request.getRequestId())
                .content(result.getContent())
                .model(request.getModel())
                .tokensUsed(result.getTokensUsed())
                .durationMs(duration)
                .metadata("provider", PROVIDER_ID)
                .metadata("mcp_metadata", result.getMetadata())
                .build();
    }

    /**
     * Build error response
     */
    private InferenceResponse buildErrorResponse(
            ProviderRequest request,
            Throwable error,
            long startTime) {
        long duration = System.currentTimeMillis() - startTime;

        return InferenceResponse.builder()
                .requestId(request.getRequestId())
                .content("Error: " + error.getMessage())
                .model(request.getModel())
                .tokensUsed(0)
                .durationMs(duration)
                .metadata("provider", PROVIDER_ID)
                .metadata("error", error.getClass().getSimpleName())
                .build();
    }

    /**
     * Shutdown and cleanup
     */
    public void shutdown() {
        LOG.info("Shutting down MCP provider");
        mcpClient.close();
        initialized = false;
    }

    // Configuration record
    private record MCPProviderConfig(
            String name,
            MCPClientConfig clientConfig) {
    }
}
