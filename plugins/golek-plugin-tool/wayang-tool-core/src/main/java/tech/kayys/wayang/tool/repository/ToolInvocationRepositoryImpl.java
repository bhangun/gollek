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
    public Uni<List<ToolInvocation>> findByRequestId(String requestId) {
        return list("requestId", requestId);
    }

    @Override
    public Uni<List<ToolInvocation>> findByToolId(String toolId) {
        return list("toolId", toolId);
    }

    @Override
    public Uni<List<ToolInvocation>> findByRequestIdAndToolId(String requestId, String toolId) {
        return list("requestId = ?1 AND toolId = ?2", requestId, toolId);
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
    public Uni<Long> countByRequestId(String requestId) {
        return count("requestId", requestId);
    }
}