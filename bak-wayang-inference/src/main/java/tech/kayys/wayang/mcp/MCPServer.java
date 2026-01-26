package tech.kayys.golek.mcp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class MCPServer implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(MCPServer.class);
    private static final ObjectMapper mapper = new ObjectMapper();
    
    private final String name;
    private final MCPConfig.MCPServerConfig config;
    private Process process;
    private BufferedReader reader;
    private BufferedWriter writer;
    private final Map<String, MCPTool> tools = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private volatile boolean running = false;
    
    public MCPServer(String name, MCPConfig.MCPServerConfig config) {
        this.name = name;
        this.config = config;
    }
    
    public void start() throws MCPException {
        if (!config.enabled()) {
            log.info("MCP server {} is disabled", name);
            return;
        }
        
        try {
            log.info("Starting MCP server: {}", name);
            
            ProcessBuilder pb = new ProcessBuilder();
            List<String> command = new ArrayList<>();
            command.add(config.command());
            if (config.args() != null) {
                command.addAll(Arrays.asList(config.args()));
            }
            
            pb.command(command);
            
            // Set environment variables
            if (config.env() != null) {
                Map<String, String> env = pb.environment();
                env.putAll(config.env());
            }
            
            pb.redirectErrorStream(true);
            
            process = pb.start();
            reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            
            running = true;
            
            // Initialize connection
            initialize();
            
            // Discover tools
            discoverTools();
            
            log.info("MCP server {} started with {} tools", name, tools.size());
            
        } catch (Exception e) {
            throw new MCPException("Failed to start MCP server: " + name, e);
        }
    }
    
    private void initialize() throws MCPException {
        try {
            MCPRequest request = new MCPRequest(
                "initialize",
                Map.of(
                    "protocolVersion", "2024-11-05",
                    "capabilities", Map.of(
                        "tools", Map.of("listChanged", true)
                    ),
                    "clientInfo", Map.of(
                        "name", "llama-platform",
                        "version", "1.0.0"
                    )
                )
            );
            
            MCPResponse response = sendRequest(request);
            
            if (response.error() != null) {
                throw new MCPException("Initialize failed: " + response.error());
            }
            
            log.debug("MCP server {} initialized", name);
            
        } catch (Exception e) {
            throw new MCPException("Failed to initialize MCP server", e);
        }
    }
    
    private void discoverTools() throws MCPException {
        try {
            MCPRequest request = new MCPRequest("tools/list", Map.of());
            MCPResponse response = sendRequest(request);
            
            if (response.error() != null) {
                throw new MCPException("Failed to list tools: " + response.error());
            }
            
            JsonNode toolsNode = mapper.valueToTree(response.result());
            JsonNode toolsArray = toolsNode.get("tools");
            
            if (toolsArray != null && toolsArray.isArray()) {
                for (JsonNode toolNode : toolsArray) {
                    String toolName = toolNode.get("name").asText();
                    String description = toolNode.has("description") ? 
                        toolNode.get("description").asText() : "";
                    JsonNode schema = toolNode.get("inputSchema");
                    
                    MCPTool tool = new MCPTool(toolName, description, schema);
                    tools.put(toolName, tool);
                    
                    log.debug("Discovered tool: {} - {}", toolName, description);
                }
            }
            
        } catch (Exception e) {
            throw new MCPException("Failed to discover tools", e);
        }
    }
    
    public String executeTool(String toolName, JsonNode arguments) throws MCPException {
        MCPTool tool = tools.get(toolName);
        if (tool == null) {
            throw new MCPException("Tool not found: " + toolName);
        }
        
        try {
            MCPRequest request = new MCPRequest(
                "tools/call",
                Map.of(
                    "name", toolName,
                    "arguments", mapper.convertValue(arguments, Map.class)
                )
            );
            
            MCPResponse response = sendRequest(request);
            
            if (response.error() != null) {
                throw new MCPException("Tool execution failed: " + response.error());
            }
            
            // Extract content from response
            JsonNode resultNode = mapper.valueToTree(response.result());
            JsonNode contentArray = resultNode.get("content");
            
            if (contentArray != null && contentArray.isArray() && contentArray.size() > 0) {
                JsonNode firstContent = contentArray.get(0);
                if (firstContent.has("text")) {
                    return firstContent.get("text").asText();
                }
            }
            
            return mapper.writeValueAsString(response.result());
            
        } catch (Exception e) {
            throw new MCPException("Failed to execute tool: " + toolName, e);
        }
    }
    
    private synchronized MCPResponse sendRequest(MCPRequest request) throws MCPException {
        if (!running || process == null || !process.isAlive()) {
            throw new MCPException("MCP server is not running");
        }
        
        try {
            // Send request
            String requestJson = mapper.writeValueAsString(request);
            writer.write(requestJson);
            writer.newLine();
            writer.flush();
            
            log.debug("Sent MCP request: {}", request.method());
            
            // Read response
            String responseLine = reader.readLine();
            if (responseLine == null) {
                throw new MCPException("No response from MCP server");
            }
            
            MCPResponse response = mapper.readValue(responseLine, MCPResponse.class);
            log.debug("Received MCP response");
            
            return response;
            
        } catch (IOException e) {
            throw new MCPException("Communication error with MCP server", e);
        }
    }
    
    public List<MCPTool> getTools() {
        return new ArrayList<>(tools.values());
    }
    
    public boolean isRunning() {
        return running && process != null && process.isAlive();
    }
    
    @Override
    public void close() {
        running = false;
        
        try {
            if (writer != null) {
                writer.close();
            }
            if (reader != null) {
                reader.close();
            }
            if (process != null) {
                process.destroy();
                process.waitFor(5, TimeUnit.SECONDS);
                if (process.isAlive()) {
                    process.destroyForcibly();
                }
            }
        } catch (Exception e) {
            log.error("Error closing MCP server: {}", name, e);
        }
        
        executor.shutdown();
        log.info("MCP server {} stopped", name);
    }
    
    public String getName() {
        return name;
    }
    
    public MCPConfig.MCPServerConfig getConfig() {
        return config;
    }
    
    private record MCPRequest(
        String method,
        Map<String, Object> params
    ) {
        public MCPRequest {
            if (method == null) throw new IllegalArgumentException("method required");
            if (params == null) params = Map.of();
        }
    }
    
    private record MCPResponse(
        Object result,
        Object error
    ) {}
}
