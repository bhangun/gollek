package tech.kayys.wayang.memory.service;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gamelan.engine.node.NodeExecutionResult;
import tech.kayys.gamelan.engine.node.DefaultNodeExecutionResult;
import tech.kayys.gamelan.engine.node.NodeExecutionTask;
import tech.kayys.gamelan.engine.node.NodeExecutionStatus;
import tech.kayys.gamelan.engine.protocol.CommunicationType;
import tech.kayys.gamelan.sdk.executor.core.AbstractWorkflowExecutor;
import tech.kayys.gamelan.sdk.executor.core.Executor;
import tech.kayys.wayang.memory.model.*;
import tech.kayys.wayang.memory.util.TextChunker;

import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * Memory-aware workflow executor that leverages semantic memory and context
 * engineering
 * to enhance task execution with relevant historical context.
 */
@Executor(executorType = "memory-aware-executor", communicationType = CommunicationType.GRPC, maxConcurrentTasks = 10)
@ApplicationScoped
public class MemoryAwareExecutor extends AbstractWorkflowExecutor {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryAwareExecutor.class);

    @Inject
    VectorMemoryStore memoryStore;

    @Inject
    EmbeddingServiceFactory embeddingFactory;

    @Inject
    ContextEngineeringService contextService;

    @Inject
    TextChunker textChunker;

    @Override
    public Uni<NodeExecutionResult> execute(NodeExecutionTask task) {
        LOG.info("Executing memory-aware task: run={}, node={}",
                task.runId().value(), task.nodeId().value());

        String namespace = buildNamespace(task);
        String taskDescription = buildTaskDescription(task);

        // Build context from memory
        ContextConfig contextConfig = ContextConfig.builder()
                .maxMemories(5)
                .memoryTypes(List.of(MemoryType.EPISODIC, MemoryType.SEMANTIC))
                .includeMetadata(true)
                .build();

        return contextService.buildContext(taskDescription, namespace, contextConfig)
                .flatMap(context -> executeWithContext(task, context))
                .flatMap(result -> storeExecutionMemory(task, result, namespace)
                        .replaceWith(result))
                .onFailure().invoke(error -> LOG.error("Memory-aware execution failed", error));
    }

    /**
     * Execute task with engineered context
     */
    private Uni<NodeExecutionResult> executeWithContext(
            NodeExecutionTask task,
            EngineerContext context) {

        LOG.debug("Executing with context: {} tokens, {} sections",
                context.getTotalTokens(), context.getSections().size());

        // Extract task-specific logic from context
        Map<String, Object> taskContext = task.context();

        // Simulate task execution with memory context
        // In real implementation, this would call actual task logic
        return Uni.createFrom().item(() -> {
            Map<String, Object> output = new HashMap<>(taskContext);

            // Add context insights
            output.put("contextSections", context.getSections().size());
            output.put("contextTokens", context.getTotalTokens());
            output.put("contextUtilization", context.getUtilization());

            // Add relevant memories summary
            List<String> relevantInsights = context.getSections().stream()
                    .filter(s -> s.getType().startsWith("memory_"))
                    .map(s -> s.getContent().substring(0, Math.min(100, s.getContent().length())))
                    .toList();
            output.put("relevantInsights", relevantInsights);

            return new DefaultNodeExecutionResult(
                    task.runId(),
                    task.nodeId(),
                    task.attempt(),
                    NodeExecutionStatus.COMPLETED,
                    output,
                    null,
                    task.token());
        });
    }

    /**
     * Store execution as memory for future reference
     */
    private Uni<String> storeExecutionMemory(
            NodeExecutionTask task,
            NodeExecutionResult result,
            String namespace) {

        LOG.debug("Storing execution memory for task: {}", task.nodeId().value());

        // Build memory content
        String memoryContent = buildMemoryContent(task, result);

        // Calculate importance based on execution characteristics
        double importance = calculateImportance(task, result);

        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        // Generate embedding and store
        return embeddingService.embed(memoryContent)
                .flatMap(embedding -> {
                    Memory memory = Memory.builder()
                            .namespace(namespace)
                            .content(memoryContent)
                            .embedding(embedding)
                            .type(MemoryType.EPISODIC)
                            .importance(importance)
                            .timestamp(Instant.now())
                            .addMetadata("runId", task.runId().value())
                            .addMetadata("nodeId", task.nodeId().value())
                            .addMetadata("attempt", task.attempt())
                            .addMetadata("status", result.status().name())
                            .addMetadata("executionTime", System.currentTimeMillis())
                            .addMetadata("accessCount", 0)
                            .build();

                    return memoryStore.store(memory);
                });
    }

    /**
     * Build namespace for memory isolation
     */
    private String buildNamespace(NodeExecutionTask task) {
        // Namespace format: workflowDef:nodeType
        Map<String, Object> context = task.context();
        String workflowDef = (String) context.getOrDefault("workflowDefinitionId", "default");
        return "workflow:" + workflowDef;
    }

    /**
     * Build task description for context retrieval
     */
    private String buildTaskDescription(NodeExecutionTask task) {
        StringBuilder description = new StringBuilder();

        Map<String, Object> context = task.context();

        description.append("Task: ").append(task.nodeId().value()).append("\n");
        description.append("Execution context:\n");

        // Add key context variables
        context.forEach((key, value) -> {
            if (value != null && !key.equals("internalState")) {
                description.append("- ").append(key).append(": ")
                        .append(value.toString().substring(0, Math.min(100, value.toString().length())))
                        .append("\n");
            }
        });

        return description.toString();
    }

    /**
     * Build memory content from execution
     */
    private String buildMemoryContent(NodeExecutionTask task, NodeExecutionResult result) {
        StringBuilder content = new StringBuilder();

        content.append("Execution Summary\n");
        content.append("=================\n");
        content.append("Node: ").append(task.nodeId().value()).append("\n");
        content.append("Status: ").append(result.status()).append("\n");
        content.append("Attempt: ").append(task.attempt()).append("\n");

        if (result.status() == NodeExecutionStatus.COMPLETED) {
            content.append("\nOutput:\n");
            result.output().forEach((key, value) -> {
                content.append("- ").append(key).append(": ").append(value).append("\n");
            });
        } else if (result.status() == NodeExecutionStatus.FAILED) {
            content.append("\nError:\n");
            if (result.error() != null) {
                content.append("Code: ").append(result.error().code()).append("\n");
                content.append("Message: ").append(result.error().message()).append("\n");
            }
        }

        return content.toString();
    }

    /**
     * Calculate importance score for memory
     * Higher importance = longer retention and higher retrieval priority
     */
    private double calculateImportance(NodeExecutionTask task, NodeExecutionResult result) {
        double importance = 0.5; // Base importance

        // Increase importance for failures (learn from mistakes)
        if (result.status() == NodeExecutionStatus.FAILED) {
            importance += 0.3;
        }

        // Increase importance for retries (indicates difficulty)
        if (task.attempt() > 1) {
            importance += 0.1 * Math.min(task.attempt(), 3);
        }

        // Increase importance for first execution
        if (task.attempt() == 1) {
            importance += 0.1;
        }

        return Math.min(1.0, importance);
    }
}