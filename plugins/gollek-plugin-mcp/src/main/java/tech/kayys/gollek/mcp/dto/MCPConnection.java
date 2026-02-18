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
 */
public class MCPConnection implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(MCPConnection.class);

    private static final List<String> SUPPORTED_PROTOCOL_VERSIONS = List.of(
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

    private Uni<Void> discoverTools() {
        if (!hasCapability("tools")) {
            return Uni.createFrom().voidItem();
        }

        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("tools/list")
                .build();

        return transport.sendRequest(request)
                .onItem().invoke(response -> {
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
                    }
                })
                .replaceWithVoid();
    }

    private Uni<Void> discoverResources() {
        if (!hasCapability("resources")) {
            return Uni.createFrom().voidItem();
        }

        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("resources/list")
                .build();

        return transport.sendRequest(request)
                .onItem().invoke(response -> {
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
                    }
                })
                .replaceWithVoid();
    }

    private Uni<Void> discoverPrompts() {
        if (!hasCapability("prompts")) {
            return Uni.createFrom().voidItem();
        }

        MCPRequest request = MCPRequest.builder()
                .id(nextRequestId())
                .method("prompts/list")
                .build();

        return transport.sendRequest(request)
                .onItem().invoke(response -> {
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
                    }
                })
                .replaceWithVoid();
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

    public boolean isConnected() {
        return transport.isConnected();
    }

    private boolean hasCapability(String capability) {
        return serverCapabilities != null && serverCapabilities.containsKey(capability);
    }

    private long nextRequestId() {
        return requestIdSequence.incrementAndGet();
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
}
