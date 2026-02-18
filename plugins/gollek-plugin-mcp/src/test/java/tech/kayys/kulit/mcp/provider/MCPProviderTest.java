package tech.kayys.gollek.mcp.provider;

import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;

import jakarta.inject.Inject;

import static org.junit.jupiter.api.Assertions.*;

@QuarkusTest
class MCPProviderTest {

    @Inject
    MCPProvider mcpProvider;

    @BeforeEach
    void setUp() {
        // Setup any required test configuration
    }

    @Test
    @DisplayName("Test MCP Provider ID")
    void testProviderId() {
        assertEquals("mcp", mcpProvider.id(), "MCP Provider should have ID 'mcp'");
    }

    @Test
    @DisplayName("Test MCP Provider Name")
    void testProviderName() {
        assertEquals("MCP Provider", mcpProvider.name(), "MCP Provider should have correct name");
    }

    @Test
    @DisplayName("Test MCP Provider Capabilities")
    void testProviderCapabilities() {
        var capabilities = mcpProvider.capabilities();
        assertNotNull(capabilities, "Capabilities should not be null");
        assertTrue(capabilities.isToolCalling(), "MCP Provider should support tools");
        assertTrue(capabilities.isMultimodal(), "MCP Provider should support multimodal");
        assertTrue(capabilities.isFunctionCalling(), "MCP Provider should support function calling");
    }

    @Test
    @DisplayName("Test MCP Provider Metadata")
    void testProviderMetadata() {
        var metadata = mcpProvider.metadata();
        assertNotNull(metadata, "Metadata should not be null");
        assertEquals("Wayang", metadata.getVendor(), "Vendor should be Wayang");
    }

    @Test
    @DisplayName("Test MCP Provider Health")
    void testProviderHealth() {
        // Note: This test may return UNHEALTHY if no MCP servers are configured
        // That's expected behavior when no servers are running
        var healthUni = mcpProvider.health();
        var health = healthUni.await().indefinitely();
        
        assertNotNull(health, "Health should not be null");
        assertNotNull(health.status(), "Health status should not be null");
        assertNotNull(health.timestamp(), "Health timestamp should not be null");
    }
}
