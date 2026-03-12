package tech.kayys.gollek.cli;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.enterprise.inject.Alternative;
import jakarta.annotation.Priority;
import tech.kayys.gollek.mcp.registry.McpRegistryEngine;
import tech.kayys.gollek.sdk.mcp.McpRegistryManager;

@ApplicationScoped
public class McpRegistryProducer {

    @Produces
    @Alternative
    @Priority(1)
    @ApplicationScoped
    public McpRegistryManager createMcpRegistryManager() {
        return new McpRegistryEngine();
    }
}
