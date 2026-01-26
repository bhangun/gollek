package tech.kayys.golek.provider.core.mcp;

import io.quarkus.runtime.Startup;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
        List<MCPClientConfig> configs = new ArrayList<>();

        config.servers().forEach((key, serverConfig) -> {
            try {
                MCPClientConfig clientConfig = buildClientConfig(key, serverConfig);
                configs.add(clientConfig);
                LOG.infof("Configured MCP server: %s (%s)",
                        serverConfig.name(), serverConfig.transport());
            } catch (Exception e) {
                LOG.errorf(e, "Failed to configure MCP server: %s", key);
            }
        });

        return configs;
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
                builder.command(serverConfig.command().get())
                        .args(serverConfig.args())
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