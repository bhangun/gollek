package tech.kayys.gollek.mcp.provider;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.quarkus.runtime.Startup;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.gollek.mcp.client.MCPClientConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Initializes MCP provider on startup if enabled.
 */
@Startup
@ApplicationScoped
public class MCPProviderInitializer {

    private static final Logger LOG = Logger.getLogger(MCPProviderInitializer.class);

    @Inject
    MCPProvider mcpProvider;

    @Inject
    MCPProviderConfiguration config;

    @Inject
    ObjectMapper objectMapper;

    @ConfigProperty(name = "wayang.inference.mcp.mcp-servers-json")
    Optional<String> mcpServersJson;

    @ConfigProperty(name = "wayang.inference.mcp.mcp-servers-json-file")
    Optional<String> mcpServersJsonFile;

    /**
     * Initialize MCP provider on startup
     */
    public void initialize() {
        if (!config.enabled()) {
            LOG.info("MCP provider is disabled");
            return;
        }

        LOG.info("Initializing MCP provider...");

        List<MCPClientConfig> serverConfigs = buildServerConfigs();

        if (serverConfigs.isEmpty()) {
            LOG.warn("No MCP servers configured");
            return;
        }

        mcpProvider.initialize(serverConfigs)
                .subscribe().with(
                        v -> LOG.info("MCP provider initialized successfully"),
                        error -> LOG.error("Failed to initialize MCP provider", error));
    }

    /**
     * Build MCP client configurations from application config
     */
    private List<MCPClientConfig> buildServerConfigs() {
        Map<String, MCPClientConfig> configsByName = new LinkedHashMap<>();

        config.servers().forEach((key, serverConfig) -> {
            try {
                MCPClientConfig clientConfig = buildClientConfig(key, serverConfig);
                configsByName.put(clientConfig.getName(), clientConfig);
                LOG.infof("Configured MCP server: %s (%s)",
                        serverConfig.name(), serverConfig.transport());
            } catch (Exception e) {
                LOG.errorf(e, "Failed to configure MCP server: %s", key);
            }
        });

        loadExternalMcpServers().forEach((name, clientConfig) -> {
            MCPClientConfig previous = configsByName.put(name, clientConfig);
            if (previous == null) {
                LOG.infof("Configured MCP server from mcpServers JSON: %s", name);
            } else {
                LOG.infof("Overriding MCP server config from mcpServers JSON: %s", name);
            }
        });

        return new ArrayList<>(configsByName.values());
    }

    private Map<String, MCPClientConfig> loadExternalMcpServers() {
        Map<String, MCPClientConfig> result = new LinkedHashMap<>();

        if (mcpServersJson.isPresent() && !mcpServersJson.get().isBlank()) {
            parseMcpServersJson(mcpServersJson.get(), "inline")
                    .forEach(result::put);
        }

        if (mcpServersJsonFile.isPresent() && !mcpServersJsonFile.get().isBlank()) {
            String filePath = mcpServersJsonFile.get().trim();
            try {
                String content = Files.readString(Path.of(filePath));
                parseMcpServersJson(content, filePath).forEach(result::put);
            } catch (Exception e) {
                LOG.errorf(e, "Failed to read mcpServers JSON file: %s", filePath);
            }
        }

        return result;
    }

    private Map<String, MCPClientConfig> parseMcpServersJson(String json, String sourceLabel) {
        Map<String, MCPClientConfig> result = new LinkedHashMap<>();
        try {
            JsonNode root = objectMapper.readTree(json);
            JsonNode serversNode = root.has("mcpServers") ? root.get("mcpServers") : root;
            if (serversNode == null || !serversNode.isObject()) {
                LOG.warnf("Ignoring invalid mcpServers JSON from %s: expected object", sourceLabel);
                return result;
            }

            serversNode.fields().forEachRemaining(entry -> {
                String key = entry.getKey();
                JsonNode serverNode = entry.getValue();
                try {
                    if (!isEnabled(serverNode)) {
                        LOG.infof("Skipping disabled MCP server '%s' from %s", key, sourceLabel);
                        return;
                    }
                    MCPClientConfig config = buildClientConfigFromJson(key, serverNode);
                    result.put(config.getName(), config);
                } catch (Exception e) {
                    LOG.errorf(e, "Failed to parse mcpServers entry '%s' from %s", key, sourceLabel);
                }
            });
        } catch (Exception e) {
            LOG.errorf(e, "Failed to parse mcpServers JSON from %s", sourceLabel);
        }
        return result;
    }

    private MCPClientConfig buildClientConfigFromJson(String key, JsonNode serverNode) {
        if (serverNode == null || !serverNode.isObject()) {
            throw new IllegalArgumentException("Server config must be a JSON object");
        }

        String transportRaw = stringValue(serverNode.get("transport"), "stdio");
        MCPClientConfig.TransportType transportType = MCPClientConfig.TransportType
                .valueOf(transportRaw.toUpperCase());

        String displayName = stringValue(serverNode.get("name"), key);
        MCPClientConfig.Builder builder = MCPClientConfig.builder()
                .name(displayName)
                .transportType(transportType);

        switch (transportType) {
            case STDIO -> {
                String command = requiredString(serverNode.get("command"),
                        "STDIO transport requires 'command'");
                builder.command(command);
                builder.args(readStringArray(serverNode.get("args")));
                builder.env(readStringMap(serverNode.get("env")));
            }
            case HTTP, WEBSOCKET -> {
                String url = requiredString(serverNode.get("url"),
                        transportType + " transport requires 'url'");
                builder.url(url);
            }
        }

        JsonNode autoReconnect = serverNode.get("autoReconnect");
        if (autoReconnect != null && autoReconnect.isBoolean()) {
            builder.autoReconnect(autoReconnect.asBoolean());
        }

        JsonNode maxReconnectAttempts = serverNode.get("maxReconnectAttempts");
        if (maxReconnectAttempts != null && maxReconnectAttempts.canConvertToInt()) {
            builder.maxReconnectAttempts(maxReconnectAttempts.asInt());
        }

        return builder.build();
    }

    private String[] readStringArray(JsonNode node) {
        if (node == null || !node.isArray()) {
            return new String[0];
        }
        List<String> values = new ArrayList<>();
        node.forEach(item -> {
            if (item != null && item.isTextual() && !item.asText().isBlank()) {
                values.add(item.asText());
            }
        });
        return values.toArray(String[]::new);
    }

    private Map<String, String> readStringMap(JsonNode node) {
        Map<String, String> values = new LinkedHashMap<>();
        if (node == null || !node.isObject()) {
            return values;
        }
        node.fields().forEachRemaining(entry -> {
            JsonNode value = entry.getValue();
            if (value != null && !value.isNull()) {
                values.put(entry.getKey(), value.asText());
            }
        });
        return values;
    }

    private String stringValue(JsonNode node, String defaultValue) {
        if (node == null || !node.isTextual() || node.asText().isBlank()) {
            return defaultValue;
        }
        return node.asText();
    }

    private String requiredString(JsonNode node, String message) {
        if (node == null || !node.isTextual() || node.asText().isBlank()) {
            throw new IllegalArgumentException(message);
        }
        return node.asText();
    }

    private boolean isEnabled(JsonNode serverNode) {
        JsonNode enabledNode = serverNode.get("enabled");
        return enabledNode == null || !enabledNode.isBoolean() || enabledNode.asBoolean();
    }

    /**
     * Build client config for a single server
     */
    private MCPClientConfig buildClientConfig(
            String key,
            MCPProviderConfiguration.MCPServerConfig serverConfig) {
        MCPClientConfig.TransportType transportType = MCPClientConfig.TransportType
                .valueOf(serverConfig.transport().toUpperCase());

        var builder = MCPClientConfig.builder()
                .name(serverConfig.name())
                .transportType(transportType);

        // Configure based on transport type
        switch (transportType) {
            case STDIO -> {
                if (serverConfig.command().isEmpty()) {
                    throw new IllegalArgumentException(
                            "STDIO transport requires 'command' to be specified");
                }
                String[] argsArray = serverConfig.args().toArray(new String[0]);
                builder.command(serverConfig.command().get())
                        .args(argsArray)
                        .env(serverConfig.env());
            }
            case HTTP, WEBSOCKET -> {
                if (serverConfig.url().isEmpty()) {
                    throw new IllegalArgumentException(
                            transportType + " transport requires 'url' to be specified");
                }
                builder.url(serverConfig.url().get());
            }
        }

        return builder.build();
    }
}
