package tech.kayys.golek.mcp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class MCPRegistry implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(MCPRegistry.class);
    private static final ObjectMapper mapper = new ObjectMapper();
    
    private final Map<String, MCPServer> servers = new ConcurrentHashMap<>();
    private final Path configPath;
    
    public MCPRegistry(Path configPath) {
        this.configPath = configPath;
    }
    
    /**
     * Load MCP servers from configuration file
     */
    public void loadFromConfig() throws MCPException {
        if (!Files.exists(configPath)) {
            log.warn("MCP config file not found: {}", configPath);
            return;
        }
        
        try {
            log.info("Loading MCP configuration from: {}", configPath);
            
            String json = Files.readString(configPath);
            MCPConfig config = mapper.readValue(json, MCPConfig.class);
            
            if (config.mcpServers() != null) {
                for (Map.Entry<String, MCPConfig.MCPServerConfig> entry : 
                        config.mcpServers().entrySet()) {
                    
                    String serverName = entry.getKey();
                    MCPConfig.MCPServerConfig serverConfig = entry.getValue();
                    
                    try {
                        registerServer(serverName, serverConfig);
                    } catch (Exception e) {
                        log.error("Failed to register MCP server: {}", serverName, e);
                    }
                }
            }
            
            log.info("Loaded {} MCP servers", servers.size());
            
        } catch (IOException e) {
            throw new MCPException("Failed to load MCP config", e);
        }
    }
    
    /**
     * Register and start an MCP server
     */
    public void registerServer(String name, MCPConfig.MCPServerConfig config) throws MCPException {
        if (servers.containsKey(name)) {
            log.warn("MCP server already registered: {}", name);
            return;
        }
        
        MCPServer server = new MCPServer(name, config);
        server.start();
        
        servers.put(name, server);
        log.info("Registered MCP server: {} with {} tools", name, server.getTools().size());
    }
    
    /**
     * Execute a tool on an MCP server
     */
    public String executeTool(String serverName, String toolName, JsonNode arguments) 
            throws MCPException {
        MCPServer server = servers.get(serverName);
        if (server == null) {
            throw new MCPException("MCP server not found: " + serverName);
        }
        
        if (!server.isRunning()) {
            throw new MCPException("MCP server is not running: " + serverName);
        }
        
        return server.executeTool(toolName, arguments);
    }
    
    /**
     * Get all available tools across all servers
     */
    public Map<String, List<MCPTool>> getAllTools() {
        Map<String, List<MCPTool>> allTools = new HashMap<>();
        
        for (Map.Entry<String, MCPServer> entry : servers.entrySet()) {
            if (entry.getValue().isRunning()) {
                allTools.put(entry.getKey(), entry.getValue().getTools());
            }
        }
        
        return allTools;
    }
    
    /**
     * Get tools from a specific server
     */
    public List<MCPTool> getTools(String serverName) {
        MCPServer server = servers.get(serverName);
        return server != null ? server.getTools() : List.of();
    }
    
    /**
     * Get all registered servers
     */
    public Map<String, MCPServerInfo> getServers() {
        Map<String, MCPServerInfo> info = new HashMap<>();
        
        for (Map.Entry<String, MCPServer> entry : servers.entrySet()) {
            MCPServer server = entry.getValue();
            info.put(entry.getKey(), new MCPServerInfo(
                entry.getKey(),
                server.getConfig().description(),
                server.isRunning(),
                server.getTools().size(),
                server.getConfig().command()
            ));
        }
        
        return info;
    }
    
    /**
     * Restart an MCP server
     */
    public void restartServer(String serverName) throws MCPException {
        MCPServer server = servers.get(serverName);
        if (server == null) {
            throw new MCPException("MCP server not found: " + serverName);
        }
        
        MCPConfig.MCPServerConfig config = server.getConfig();
        server.close();
        servers.remove(serverName);
        
        registerServer(serverName, config);
    }
    
    /**
     * Stop and unregister an MCP server
     */
    public void unregisterServer(String serverName) {
        MCPServer server = servers.remove(serverName);
        if (server != null) {
            server.close();
            log.info("Unregistered MCP server: {}", serverName);
        }
    }
    
    @Override
    public void close() {
        log.info("Shutting down MCP registry...");
        
        for (MCPServer server : servers.values()) {
            try {
                server.close();
            } catch (Exception e) {
                log.error("Error closing MCP server: {}", server.getName(), e);
            }
        }
        
        servers.clear();
        log.info("MCP registry shutdown complete");
    }
    
    public record MCPServerInfo(
        String name,
        String description,
        boolean running,
        int toolCount,
        String command
    ) {}
}
