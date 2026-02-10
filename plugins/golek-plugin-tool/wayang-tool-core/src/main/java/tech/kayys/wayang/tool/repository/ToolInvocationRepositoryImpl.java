package tech.kayys.wayang.tool.repository;

import io.quarkus.hibernate.reactive.panache.PanacheRepository;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.entity.ToolInvocation;

import java.util.List;
import java.util.UUID;

@ApplicationScoped
public class ToolInvocationRepositoryImpl implements PanacheRepository<ToolInvocation>, ToolInvocationRepository {

    @Override
    public Uni<List<ToolInvocation>> getAllInvocations() {
        return listAll();
    }

    @Override
    public Uni<List<ToolInvocation>> findByTenantId(String tenantId) {
        return list("tenantId", tenantId);
    }

    @Override
    public Uni<List<ToolInvocation>> findByToolId(String toolId) {
        return list("toolId", toolId);
    }

    @Override
    public Uni<List<ToolInvocation>> findByTenantIdAndToolId(String tenantId, String toolId) {
        return list("tenantId = ?1 AND toolId = ?2", tenantId, toolId);
    }

    @Override
    public Uni<ToolInvocation> findById(UUID invocationId) {
        return find("invocationId", invocationId).firstResult();
    }

    @Override
    public Uni<ToolInvocation> save(ToolInvocation invocation) {
        return persist(invocation);
    }

    @Override
    public Uni<ToolInvocation> update(ToolInvocation invocation) {
        return persist(invocation);
    }

    @Override
    public Uni<Boolean> deleteById(UUID invocationId) {
        return delete("invocationId", invocationId).map(deleted -> deleted > 0);
    }

    @Override
    public Uni<List<ToolInvocation>> searchInvocations(String query, Object... params) {
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