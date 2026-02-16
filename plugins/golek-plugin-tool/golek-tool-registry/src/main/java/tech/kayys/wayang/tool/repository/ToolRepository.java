package tech.kayys.wayang.tool.repository;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.tool.entity.McpTool;

import java.util.List;

public interface ToolRepository {

    Uni<List<McpTool>> listAllTools();

    Uni<List<McpTool>> findByRequestId(String requestId);

    Uni<List<McpTool>> findByNamespace(String namespace);

    Uni<List<McpTool>> findByRequestIdAndNamespace(String requestId, String namespace);

    Uni<McpTool> findById(String toolId);

    Uni<McpTool> findByRequestIdAndToolId(String requestId, String toolId);

    Uni<McpTool> save(McpTool tool);

    Uni<McpTool> update(McpTool tool);

    Uni<Boolean> deleteById(String toolId);

    Uni<List<McpTool>> searchTools(String query, Object... params);

    Uni<Long> count();

    Uni<Long> countByRequestId(String requestId);
}
