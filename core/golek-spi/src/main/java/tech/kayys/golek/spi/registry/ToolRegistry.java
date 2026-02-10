package tech.kayys.golek.spi.registry;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.spi.tool.Tool;

import java.util.List;

/**
 * Registry for discovering and retrieving tools.
 */
public interface ToolRegistry {

    /**
     * Register a tool.
     *
     * @param tool the tool to register
     */
    void register(Tool tool);

    /**
     * Get tool by ID.
     *
     * @param id tool ID
     * @return Uni Tool or failure if not found
     */
    Uni<Tool> getTool(String id);

    /**
     * List all available tools.
     *
     * @return Uni List of tools
     */
    Uni<List<Tool>> listTools();
}