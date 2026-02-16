package tech.kayys.wayang.tool.repository;

import io.quarkus.hibernate.reactive.panache.PanacheRepository;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.entity.McpTool;

import java.util.List;

@ApplicationScoped
public class ToolRepositoryImpl implements PanacheRepository<McpTool>, ToolRepository {

    @Override
    public Uni<List<McpTool>> listAllTools() {
        return listAll();
    }

    @Override
    public Uni<List<McpTool>> findByRequestId(String requestId) {
        return find("requestId", requestId).list();
    }

    @Override
    public Uni<List<McpTool>> findByNamespace(String namespace) {
        return find("namespace", namespace).list();
    }

    @Override
    public Uni<List<McpTool>> findByRequestIdAndNamespace(String requestId, String namespace) {
        return find("requestId = ?1 AND namespace = ?2", requestId, namespace).list();
    }

    @Override
    public Uni<McpTool> findById(String toolId) {
        return find("toolId", toolId).firstResult();
    }

    @Override
    public Uni<McpTool> findByRequestIdAndToolId(String requestId, String toolId) {
        return find("requestId = ?1 AND toolId = ?2", requestId, toolId).firstResult();
    }

    @Override
    public Uni<McpTool> save(McpTool tool) {
        return persist(tool);
    }

    @Override
    public Uni<McpTool> update(McpTool tool) {
        // In Panache, updates are performed by calling persist on an existing entity
        return persist(tool);
    }

    @Override
    public Uni<Boolean> deleteById(String toolId) {
        return delete("toolId", toolId).map(deleted -> deleted > 0);
    }

    @Override
    public Uni<List<McpTool>> searchTools(String query, Object... params) {
        return list(query, params);
    }

    @Override
    public Uni<Long> count() {
        return PanacheRepository.super.count();
    }

    @Override
    public Uni<Long> countByRequestId(String requestId) {
        return count("requestId", requestId);
    }
}