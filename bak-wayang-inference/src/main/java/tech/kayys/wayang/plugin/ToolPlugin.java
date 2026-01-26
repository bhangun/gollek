package tech.kayys.golek.plugin;

import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;

import tech.kayys.golek.model.Tool;

public interface ToolPlugin extends Plugin {
    
    /**
     * Get all tools provided by this plugin
     */
    List<Tool> getTools();
    
    /**
     * Execute a tool
     */
    String executeTool(String toolName, JsonNode arguments) throws PluginException;
    
    /**
     * Check if tool execution is allowed
     */
    default boolean isToolAllowed(String toolName) {
        return true;
    }
    
    /**
     * Called before tool execution
     */
    default void onToolExecutionStart(String toolName, JsonNode arguments) {}
    
    /**
     * Called after tool execution
     */
    default void onToolExecutionComplete(String toolName, String result) {}
}
