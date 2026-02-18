package tech.kayys.gollek.mcp.provider;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.mcp.client.MCPClient;
import tech.kayys.gollek.mcp.client.MCPClientConfig;
import tech.kayys.gollek.mcp.dto.MCPInferenceContext;
import tech.kayys.gollek.mcp.dto.MCPInferenceResult;
import tech.kayys.gollek.mcp.dto.MCPToolResult;
import tech.kayys.gollek.mcp.prompt.MCPPromptProvider;
import tech.kayys.gollek.mcp.resource.MCPResourceProvider;
import tech.kayys.gollek.mcp.tool.MCPToolExecutor;
import tech.kayys.gollek.mcp.tool.MCPToolRegistry;
import tech.kayys.gollek.mcp.dto.MCPPromptResult;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderHealth;

import org.jboss.logging.Logger;

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
                .toolCalling(true)
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

        LOG.infof("Initializing MCP provider with %d server(s)", serverConfigs.size());

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
    public String name() {
        return "MCP Provider";
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(name())
                .vendor("Wayang")
                .version("1.0.0")
                .homepage("https://github.com/kayys/wayang")
                .build();
    }


    @Override
    public void initialize(tech.kayys.gollek.spi.provider.ProviderConfig config) throws ProviderException.ProviderInitializationException {
        // Already handled by MCPProviderInitializer, but we can add additional config here if needed
        LOG.info("MCP Provider initialized with config");
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        // MCP provider can support any model that the connected MCP servers support
        // For now, we'll return true if we have at least one active connection
        return initialized && !mcpClient.getActiveConnections().isEmpty();
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

    @Override
    public Uni<ProviderHealth> health() {
        boolean hasActiveConnections = !mcpClient.getActiveConnections().isEmpty();
        
        ProviderHealth.Status status = hasActiveConnections ? 
                ProviderHealth.Status.HEALTHY : ProviderHealth.Status.UNHEALTHY;
                
        return Uni.createFrom().item(ProviderHealth.builder()
                .status(status)
                .timestamp(java.time.Instant.now())
                .details(Map.of(
                        "active_connections", mcpClient.getActiveConnections().size(),
                        "initialized", initialized
                ))
                .build());
    }


    /**
     * Process inference request through MCP workflow
     */
    private Uni<MCPInferenceResult> processRequest(ProviderRequest request) {
        var context = MCPInferenceContext.builder()
                .requestId(request.getRequestId())
                .model(request.getModel())
                .messages(convertMessages(request.getMessages()))
                .parameters(request.getParameters())
                .build();

        // Step 1: Check for tool calls in request
        return extractToolCalls(context)
                .onItem().transformToUni(toolCalls -> {
                    if (!toolCalls.isEmpty()) {
                        // Execute tools
                        return executeTools(toolCalls)
                                .onItem().transformToUni(toolResults -> processToolResults(toolResults));
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
            Object toolsParam = context.getParameters().get("tools");
            if (!(toolsParam instanceof List<?> tools) || tools.isEmpty()) {
                return toolCalls;
            }

            tools.forEach(tool -> {
                if (tool instanceof Map<?, ?> toolData) {
                    String toolName = asString(toolData.get("name"));
                    Map<String, Object> arguments = asObjectMap(toolData.get("arguments"));
                    if (toolName != null) {
                        toolCalls.put(toolName, arguments);
                    }
                }
            });

            return toolCalls;
        });
    }

    /**
     * Execute multiple tools in parallel
     */
    private Uni<Map<String, MCPToolResult>> executeTools(
            Map<String, Map<String, Object>> toolCalls) {
        LOG.debugf("Executing %d MCP tools", toolCalls.size());
        return toolExecutor.executeTools(toolCalls)
                .onItem().invoke(results -> LOG.debugf("Executed %d tools, %d succeeded",
                        results.size(),
                        results.values().stream().filter(MCPToolResult::isSuccess).count()));
    }

    /**
     * Convert SPI messages to MCP messages
     */
    private List<tech.kayys.gollek.mcp.dto.Message> convertMessages(List<tech.kayys.gollek.spi.Message> messages) {
        return messages.stream()
                .map(msg -> new tech.kayys.gollek.mcp.dto.Message(
                        msg.getRole().name(),
                        msg.getContent(),
                        null)) // No additional properties in the SPI Message
                .toList();
    }

    /**
     * Process tool results and build response
     */
    private Uni<MCPInferenceResult> processToolResults(
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
    private Uni<MCPPromptResult> checkPromptExecution(
            MCPInferenceContext context) {
        String promptName = asString(context.getParameters().get("prompt"));
        if (promptName == null || promptName.isBlank()) {
            return Uni.createFrom().nullItem();
        }

        return promptProvider.executePrompt(
                promptName,
                asStringMap(context.getParameters().get("prompt_arguments")));
    }

    /**
     * Check if request needs resource access
     */
    private Uni<String> checkResourceAccess(MCPInferenceContext context) {
        List<String> resourceUris = asStringList(context.getParameters().get("resources"));
        if (resourceUris.isEmpty()) {
            return Uni.createFrom().nullItem();
        }

        return resourceProvider.readResources(resourceUris)
                .onItem().transform(contents -> contents.values().stream()
                        .map(content -> content.getContentAsString())
                        .filter(text -> text != null && !text.isBlank())
                        .reduce((a, b) -> a + "\n\n" + b)
                        .orElse(null));
    }

    private String asString(Object value) {
        return value instanceof String str ? str : null;
    }

    private Map<String, Object> asObjectMap(Object value) {
        if (!(value instanceof Map<?, ?> map) || map.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> result = new HashMap<>();
        map.forEach((key, mapValue) -> {
            if (key != null) {
                result.put(String.valueOf(key), mapValue);
            }
        });
        return result;
    }

    private Map<String, String> asStringMap(Object value) {
        if (!(value instanceof Map<?, ?> map) || map.isEmpty()) {
            return Map.of();
        }
        Map<String, String> result = new HashMap<>();
        map.forEach((key, mapValue) -> {
            if (key != null && mapValue != null) {
                result.put(String.valueOf(key), String.valueOf(mapValue));
            }
        });
        return result;
    }

    private List<String> asStringList(Object value) {
        if (!(value instanceof List<?> list) || list.isEmpty()) {
            return List.of();
        }
        return list.stream()
                .filter(String.class::isInstance)
                .map(String.class::cast)
                .toList();
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
