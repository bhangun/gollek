package tech.kayys.wayang.tool.service;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.entity.McpTool;

@ApplicationScoped
public class ToolRegistry {

    public Uni<McpTool> resolveTool(String toolId, String tenantId) {
        // Stub implementation
        return Uni.createFrom().nullItem();
    }
}
