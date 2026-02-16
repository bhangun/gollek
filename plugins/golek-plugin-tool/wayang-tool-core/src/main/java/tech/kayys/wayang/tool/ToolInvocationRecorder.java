package tech.kayys.wayang.tool;

import java.util.UUID;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.wayang.tool.entity.ToolInvocation;
import tech.kayys.wayang.tool.repository.ToolInvocationRepository;
import tech.kayys.wayang.tool.dto.ToolExecutionRequest;
import tech.kayys.wayang.tool.dto.ToolExecutionResult;

/**
 * Tool invocation recorder for audit
 */
@ApplicationScoped
public class ToolInvocationRecorder {

    @Inject
    ToolInvocationRepository toolInvocationRepository;

    public Uni<Void> record(
            ToolExecutionRequest request,
            ToolExecutionResult result,
            java.time.Instant startTime) {

        return io.quarkus.hibernate.reactive.panache.Panache.withTransaction(() -> {
            ToolInvocation invocation = new ToolInvocation();
            invocation.setInvocationId(UUID.randomUUID());
            invocation.setRequestId(request.requestId());
            invocation.setToolId(request.toolId());
            invocation.setWorkflowRunId(request.workflowRunId());
            invocation.setAgentId(request.agentId());
            invocation.setUserId(request.userId());
            invocation.setArguments(request.arguments());
            invocation.setResult(result.output());
            invocation.setStatus(result.status());
            invocation.setErrorMessage(result.errorMessage());
            invocation.setExecutionTimeMs(result.executionTimeMs());
            invocation.setInvokedAt(startTime);
            invocation.setCompletedAt(java.time.Instant.now());

            return toolInvocationRepository.save(invocation).replaceWithVoid();
        });
    }
}