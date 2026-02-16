package tech.kayys.wayang.tool.repository;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.tool.entity.ToolInvocation;

import java.util.List;
import java.util.UUID;

public interface ToolInvocationRepository {

    Uni<List<ToolInvocation>> getAllInvocations();

    Uni<List<ToolInvocation>> findByRequestId(String requestId);

    Uni<List<ToolInvocation>> findByToolId(String toolId);

    Uni<List<ToolInvocation>> findByRequestIdAndToolId(String requestId, String toolId);

    Uni<ToolInvocation> findById(UUID invocationId);

    Uni<ToolInvocation> save(ToolInvocation invocation);

    Uni<ToolInvocation> update(ToolInvocation invocation);

    Uni<Boolean> deleteById(UUID invocationId);

    Uni<List<ToolInvocation>> searchInvocations(String query, Object... params);

    Uni<Long> count();

    Uni<Long> countByRequestId(String requestId);
}