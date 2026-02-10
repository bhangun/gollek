package tech.kayys.wayang.tool.impl;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.spi.Tool;
import tech.kayys.wayang.tool.spi.ToolRegistry;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-memory implementation of ToolRegistry.
 */
@ApplicationScoped
public class InMemoryToolRegistry implements ToolRegistry {

    private final Map<String, Tool> tools = new ConcurrentHashMap<>();

    @Override
    public void register(Tool tool) {
        tools.put(tool.id(), tool);
    }

    @Override
    public Uni<Tool> getTool(String id) {
        Tool tool = tools.get(id);
        if (tool == null) {
            return Uni.createFrom().failure(new IllegalArgumentException("Tool not found: " + id));
        }
        return Uni.createFrom().item(tool);
    }

    @Override
    public Uni<List<Tool>> listTools() {
        return Uni.createFrom().item(new ArrayList<>(tools.values()));
    }
}
