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
    public Uni<List<McpTool>> findByTenantId(String tenantId) {
        return find("tenantId", tenantId).list();
    }

    @Override
    public Uni<List<McpTool>> findByNamespace(String namespace) {
        return find("namespace", namespace).list();
    }

    @Override
    public Uni<List<McpTool>> findByTenantIdAndNamespace(String tenantId, String namespace) {
        return find("tenantId = ?1 AND namespace = ?2", tenantId, namespace).list();
    }

    @Override
    public Uni<McpTool> findById(String toolId) {
        return find("toolId", toolId).firstResult();
    }

    @Override
    public Uni<McpTool> findByTenantIdAndToolId(String tenantId, String toolId) {
        return find("tenantId = ?1 AND toolId = ?2", tenantId, toolId).firstResult();
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
    public Uni<Long> countByTenantId(String tenantId) {
        return count("tenantId", tenantId);
    }
}