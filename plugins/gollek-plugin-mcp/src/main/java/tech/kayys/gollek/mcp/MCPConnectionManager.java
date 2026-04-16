package tech.kayys.gollek.mcp;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.mcp.client.MCPClient;

import java.util.List;

/**
 * Component for managing MCP connections.
 */
@ApplicationScoped
public class MCPConnectionManager {

    private static final Logger LOG = Logger.getLogger(MCPConnectionManager.class);

    @Inject
    MCPClient mcpClient;

    public void initializeConnections() {
        LOG.info("Initializing MCP connections...");
        // Initialize configured connections
    }

    public void shutdownConnections() {
        LOG.info("Shutting down MCP connections");
        mcpClient.close();
    }
}