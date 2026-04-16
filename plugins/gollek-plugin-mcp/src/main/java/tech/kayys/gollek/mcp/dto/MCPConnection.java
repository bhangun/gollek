package tech.kayys.gollek.mcp.dto;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Uni;

import org.jboss.logging.Logger;

import tech.kayys.gollek.mcp.client.MCPClientConfig;
import tech.kayys.gollek.mcp.client.MCPTransport;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Represents a connection to a single MCP server.
 * Manages tools, resources, and prompts exposed by the server.
 * 
 * <p>Compliant with MCP specification 2025-11-25.
 * Supports:
 * <ul>
 *   <li>Pagination via cursors on all list endpoints</li>
 *   <li>Resource subscriptions</li>
 *   <li>Roots capability (server→client)</li>
 *   <li>Sampling capability (server→client)</li>
 *   <li>Logging capability</li>
 *   <li>Completions capability</li>
 * </ul>
 */
public class MCPConnection implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(MCPConnection.class);

    private static final List<String> SUPPORTED_PROTOCOL_VERSIONS = List.of(
            "2025-11-25",
            "2025-11-05",
            "2025-03-26",
            "2024-11-05");
    private static final String PREFERRED_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS.get(0);

    private final MCPClientConfig config;
    private final MCPTransport transport;
    private final ObjectMapper objectMapper;
    private final AtomicLong requestIdSequence = new AtomicLong(System.currentTimeMillis());

    private final Map<String, MCPTool> tools = new ConcurrentHashMap<>();
    private final Map<String, MCPResource> resources = new ConcurrentHashMap<>();
    private final Map<String, MCPPrompt> prompts = new ConcurrentHashMap<>();

    private Map<String, Object> serverInfo;
    private Map<String, Object> serverCapabilities;
    private String negotiatedProtocolVersion;
    private String serverInstructions;

    /** Callback handlers for server→client requests */
    private volatile RootsProvider rootsProvider;
    private volatile SamplingHandler samplingHandler;

    public MCPConnection(
            MCPClientConfig config,
            MCPTransport transport,
            ObjectMapper objectMapper) {
        this.config = config;
        this.transport = transport;
        this.objectMapper = objectMapper;
    }

    /**
     * Connect and initialize
     */
    public Uni<Void> connect() {
        return transport.connect()
                .onItem().transformToUni(v -> initialize())
                .onItem().transformToUni(v -> discoverCapabilities());
    }

    /**
     * Initialize connection with server
     */
    private Uni<Void> initialize() {
        Map<String, Object> params = Map.of(
                "protocolVersion", PREFERRED_PROTOCOL_VERSION,
                "capabilities", Map.of(
                        "roots", Map.of("listChanged", true),
                        "sampling", Map.of()),
                "clientInfo", Map.of(
                        "name", "wayang-inference-server",
                        "version", "1.0.0"));

        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("initialize")
                .params(params)
                .build();

        return transport.sendRequest(request)
                .onItem().transformToUni(response -> {
                    if (!response.isSuccess()) {
                        return Uni.createFrom().failure(
                                new IllegalStateException("MCP initialize failed: " + response.getError()));
                    }

                    @SuppressWarnings("unchecked")
                    Map<String, Object> result = (Map<String, Object>) response.getResult();
                    if (result == null) {
                        return Uni.createFrom().failure(
                                new IllegalStateException("MCP initialize returned null result"));
                    }

                    Object protocolVersion = result.get("protocolVersion");
                    if (protocolVersion instanceof String version) {
                        if (!SUPPORTED_PROTOCOL_VERSIONS.contains(version)) {
                            return Uni.createFrom().failure(
                                    new IllegalStateException("Unsupported MCP protocol version: " + version));
                        }
                        negotiatedProtocolVersion = version;
                    } else {
                        LOG.warn("MCP initialize response missing protocolVersion");
                    }

                    @SuppressWarnings("unchecked")
                    Map<String, Object> info = (Map<String, Object>) result.get("serverInfo");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> capabilities = (Map<String, Object>) result.get("capabilities");
                    serverInfo = info != null ? info : Map.of();
                    serverCapabilities = capabilities != null ? capabilities : Map.of();
                    serverInstructions = (String) result.get("instructions");
                    
                    LOG.infof("MCP server initialized: %s (protocol=%s)",
                            serverInfo.getOrDefault("name", config.getName()),
                            negotiatedProtocolVersion != null ? negotiatedProtocolVersion : "unknown");

                    return transport.sendNotification("notifications/initialized", Map.of());
                })
                .replaceWithVoid();
    }

    /**
     * Discover tools, resources, and prompts
     */
    private Uni<Void> discoverCapabilities() {
        return Uni.combine().all().unis(
                discoverTools(),
                discoverResources(),
                discoverPrompts()).discardItems();
    }

    /**
     * Discover tools with pagination support.
     */
    private Uni<Void> discoverTools() {
        if (!hasCapability("tools")) {
            return Uni.createFrom().voidItem();
        }

        return discoverToolsWithCursor(null);
    }

    private Uni<Void> discoverToolsWithCursor(String cursor) {
        MCPRequest.Builder requestBuilder = MCPRequest.builder()
                .id(nextRequestId())
                .method("tools/list");
        
        if (cursor != null) {
            requestBuilder.param("cursor", cursor);
        }

        return transport.sendRequest(requestBuilder.build())
                .onItem().transformToUni(response -> {
                    if (response.isSuccess()) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> result = (Map<String, Object>) response.getResult();
                        @SuppressWarnings("unchecked")
                        List<Map<String, Object>> toolList = (List<Map<String, Object>>) result.get("tools");

                        if (toolList != null) {
                            toolList.forEach(toolData -> {
                                MCPTool tool = MCPTool.fromMap(toolData);
                                tools.put(tool.getName(), tool);
                            });
                            LOG.infof("Discovered %d tools from MCP server", tools.size());
                        }

                        // Check for pagination cursor
                        String nextCursor = (String) result.get("nextCursor");
                        if (nextCursor != null && !nextCursor.isEmpty()) {
                            // Continue fetching next page
                            return discoverToolsWithCursor(nextCursor);
                        }
                    }
                    return Uni.createFrom().voidItem();
                });
    }

    /**
     * Discover resources with pagination support.
     */
    private Uni<Void> discoverResources() {
        if (!hasCapability("resources")) {
            return Uni.createFrom().voidItem();
        }

        return discoverResourcesWithCursor(null);
    }

    private Uni<Void> discoverResourcesWithCursor(String cursor) {
        MCPRequest.Builder requestBuilder = MCPRequest.builder()
                .id(nextRequestId())
                .method("resources/list");
        
        if (cursor != null) {
            requestBuilder.param("cursor", cursor);
        }

        return transport.sendRequest(requestBuilder.build())
                .onItem().transformToUni(response -> {
                    if (response.isSuccess()) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> result = (Map<String, Object>) response.getResult();
                        @SuppressWarnings("unchecked")
                        List<Map<String, Object>> resourceList = (List<Map<String, Object>>) result.get("resources");

                        if (resourceList != null) {
                            resourceList.forEach(resourceData -> {
                                MCPResource resource = MCPResource.fromMap(resourceData);
                                resources.put(resource.getUri(), resource);
                            });
                            LOG.infof("Discovered %d resources from MCP server", resources.size());
                        }

                        // Check for pagination cursor
                        String nextCursor = (String) result.get("nextCursor");
                        if (nextCursor != null && !nextCursor.isEmpty()) {
                            return discoverResourcesWithCursor(nextCursor);
                        }
                    }
                    return Uni.createFrom().voidItem();
                });
    }

    /**
     * Discover prompts with pagination support.
     */
    private Uni<Void> discoverPrompts() {
        if (!hasCapability("prompts")) {
            return Uni.createFrom().voidItem();
        }

        return discoverPromptsWithCursor(null);
    }

    private Uni<Void> discoverPromptsWithCursor(String cursor) {
        MCPRequest.Builder requestBuilder = MCPRequest.builder()
                .id(nextRequestId())
                .method("prompts/list");
        
        if (cursor != null) {
            requestBuilder.param("cursor", cursor);
        }

        return transport.sendRequest(requestBuilder.build())
                .onItem().transformToUni(response -> {
                    if (response.isSuccess()) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> result = (Map<String, Object>) response.getResult();
                        @SuppressWarnings("unchecked")
                        List<Map<String, Object>> promptList = (List<Map<String, Object>>) result.get("prompts");

                        if (promptList != null) {
                            promptList.forEach(promptData -> {
                                MCPPrompt prompt = MCPPrompt.fromMap(promptData);
                                prompts.put(prompt.getName(), prompt);
                            });
                            LOG.infof("Discovered %d prompts from MCP server", prompts.size());
                        }

                        // Check for pagination cursor
                        String nextCursor = (String) result.get("nextCursor");
                        if (nextCursor != null && !nextCursor.isEmpty()) {
                            return discoverPromptsWithCursor(nextCursor);
                        }
                    }
                    return Uni.createFrom().voidItem();
                });
    }

    /**
     * Call a tool
     */
    public Uni<MCPResponse> callTool(String toolName, Map<String, Object> arguments) {
        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("tools/call")
                .param("name", toolName)
                .param("arguments", arguments)
                .build();

        return transport.sendRequest(request);
    }

    /**
     * Read a resource
     */
    public Uni<MCPResponse> readResource(String uri) {
        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("resources/read")
                .param("uri", uri)
                .build();

        return transport.sendRequest(request);
    }

    /**
     * Subscribe to resource updates
     */
    public Uni<Void> subscribeResource(String uri) {
        if (!hasCapability("resources") || !supportsResourceSubscribe()) {
            return Uni.createFrom().failure(
                    new IllegalStateException("Server does not support resource subscriptions"));
        }

        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("resources/subscribe")
                .param("uri", uri)
                .build();

        return transport.sendRequest(request)
                .onItem().transform(response -> {
                    if (!response.isSuccess()) {
                        throw new IllegalStateException("Resource subscription failed: " + response.getError());
                    }
                    return null;
                })
                .replaceWithVoid();
    }

    /**
     * Unsubscribe from resource updates
     */
    public Uni<Void> unsubscribeResource(String uri) {
        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("resources/unsubscribe")
                .param("uri", uri)
                .build();

        return transport.sendRequest(request)
                .onItem().transform(response -> {
                    if (!response.isSuccess()) {
                        throw new IllegalStateException("Resource unsubscription failed: " + response.getError());
                    }
                    return null;
                })
                .replaceWithVoid();
    }

    /**
     * Get a prompt
     */
    public Uni<MCPResponse> getPrompt(String promptName, Map<String, String> arguments) {
        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("prompts/get")
                .param("name", promptName)
                .param("arguments", arguments != null ? arguments : Map.of())
                .build();

        return transport.sendRequest(request);
    }

    /**
     * Request argument completion
     */
    public Uni<MCPResponse> complete(String refType, String refName, String refUri, 
                                     String argumentName, String argumentValue) {
        Map<String, Object> ref = new HashMap<>();
        ref.put("type", refType);
        if (refName != null) ref.put("name", refName);
        if (refUri != null) ref.put("uri", refUri);

        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("completion/complete")
                .param("ref", ref)
                .param("argument", Map.of(
                        "name", argumentName,
                        "value", argumentValue))
                .build();

        return transport.sendRequest(request);
    }

    /**
     * Set logging level
     */
    public Uni<Void> setLoggingLevel(String level) {
        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("logging/setLevel")
                .param("level", level)
                .build();

        return transport.sendRequest(request)
                .onItem().transform(response -> {
                    if (!response.isSuccess()) {
                        throw new IllegalStateException("Failed to set logging level: " + response.getError());
                    }
                    return null;
                })
                .replaceWithVoid();
    }

    /**
     * Ping the server
     */
    public Uni<Void> ping() {
        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("ping")
                .build();

        return transport.sendRequest(request)
                .onItem().transform(response -> {
                    if (!response.isSuccess()) {
                        throw new IllegalStateException("Ping failed: " + response.getError());
                    }
                    return null;
                })
                .replaceWithVoid();
    }

    // Getters
    public MCPClientConfig getConfig() {
        return config;
    }

    public Map<String, MCPTool> getTools() {
        return Collections.unmodifiableMap(tools);
    }

    public Map<String, MCPResource> getResources() {
        return Collections.unmodifiableMap(resources);
    }

    public Map<String, MCPPrompt> getPrompts() {
        return Collections.unmodifiableMap(prompts);
    }

    public Map<String, Object> getServerInfo() {
        return serverInfo;
    }

    public Map<String, Object> getServerCapabilities() {
        return serverCapabilities;
    }

    public String getServerInstructions() {
        return serverInstructions;
    }

    public boolean isConnected() {
        return transport.isConnected();
    }

    public String getNegotiatedProtocolVersion() {
        return negotiatedProtocolVersion;
    }

    private boolean hasCapability(String capability) {
        return serverCapabilities != null && serverCapabilities.containsKey(capability);
    }

    @SuppressWarnings("unchecked")
    private boolean supportsResourceSubscribe() {
        if (!hasCapability("resources")) return false;
        Map<String, Object> resourcesCap = (Map<String, Object>) serverCapabilities.get("resources");
        return resourcesCap != null && Boolean.TRUE.equals(resourcesCap.get("subscribe"));
    }

    private long nextRequestId() {
        return requestIdSequence.incrementAndGet();
    }

    /**
     * Set roots provider callback
     */
    public void setRootsProvider(RootsProvider provider) {
        this.rootsProvider = provider;
    }

    /**
     * Get roots provider
     */
    public RootsProvider getRootsProvider() {
        return rootsProvider;
    }

    /**
     * Set sampling handler callback
     */
    public void setSamplingHandler(SamplingHandler handler) {
        this.samplingHandler = handler;
    }

    /**
     * Get sampling handler
     */
    public SamplingHandler getSamplingHandler() {
        return samplingHandler;
    }

    /**
     * Disconnect from server
     */
    public Uni<Void> disconnect() {
        return transport.disconnect()
                .onItem().invoke(() -> {
                    tools.clear();
                    resources.clear();
                    prompts.clear();
                });
    }

    @Override
    public void close() {
        disconnect().await().indefinitely();
        transport.close();
    }

    /**
     * Roots provider callback interface.
     * Called when server requests roots/list.
     */
    @FunctionalInterface
    public interface RootsProvider {
        /**
         * Provide list of exposed file system roots.
         * MUST prompt user for consent before exposing roots.
         * MUST validate URIs against path traversal.
         */
        Uni<List<RootInfo>> provideRoots();
    }

    /**
     * Sampling handler callback interface.
     * Called when server requests sampling/createMessage.
     * Per spec: MUST require explicit user approval.
     */
    @FunctionalInterface
    public interface SamplingHandler {
        /**
         * Handle a sampling request from the server.
         */
        Uni<MCPResponse> handleSampling(Map<String, Object> params);
    }
}
