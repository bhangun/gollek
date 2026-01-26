package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Represents a connection to a single MCP server.
 * Manages tools, resources, and prompts exposed by the server.
 */
public class MCPConnection implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(MCPConnection.class);

    private final MCPClientConfig config;
    private final MCPTransport transport;
    private final ObjectMapper objectMapper;

    private final Map<String, MCPTool> tools = new ConcurrentHashMap<>();
    private final Map<String, MCPResource> resources = new ConcurrentHashMap<>();
    private final Map<String, MCPPrompt> prompts = new ConcurrentHashMap<>();

    private Map<String, Object> serverInfo;
    private Map<String, Object> serverCapabilities;

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
                "protocolVersion", "2024-11-05",
                "capabilities", Map.of(
                        "roots", Map.of("listChanged", true),
                        "sampling", Map.of()),
                "clientInfo", Map.of(
                        "name", "wayang-inference-server",
                        "version", "1.0.0"));

        MCPRequest request = MCPRequest.builder()
                .id(System.currentTimeMillis())
                .method("initialize")
                .params(params)
                .build();

        return transport.sendRequest(request)
                .onItem().invoke(response -> {
                    if (response.isSuccess()) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> result = (Map<String, Object>) response.getResult();
                        serverInfo = (Map<String, Object>) result.get("serverInfo");
                        serverCapabilities = (Map<String, Object>) result.get("capabilities");
                        LOG.infof("MCP server initialized: %s", serverInfo.get("name"));
                    }
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
                .id(System.currentTimeMillis())
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
                .id(System.currentTimeMillis())
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
                .id(System.currentTimeMillis())
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
                .id(System.currentTimeMillis())
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
                .id(System.currentTimeMillis())
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
                .id(System.currentTimeMillis())
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