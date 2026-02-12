package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.model.*;
import tech.kayys.wayang.memory.context.ScoredMemory;


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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Specialized executor for querying and retrieving memories.
 * Provides advanced query capabilities:
 * - Semantic search
 * - Temporal queries
 * - Metadata filtering
 * - Cross-namespace search
 */
@Executor(executorType = "memory-query", communicationType = tech.kayys.gamelan.engine.protocol.CommunicationType.GRPC, maxConcurrentTasks = 20)
@ApplicationScoped
public class MemoryQueryExecutor extends AbstractWorkflowExecutor {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryQueryExecutor.class);

    @Inject
    VectorMemoryStore memoryStore;

    @Inject
    EmbeddingServiceFactory embeddingFactory;

    @Inject
    ContextEngineeringService contextService;

    @Override
    public Uni<NodeExecutionResult> execute(NodeExecutionTask task) {
        Map<String, Object> context = task.context();

        String query = (String) context.get("query");
        String namespace = (String) context.getOrDefault("namespace", "default");
        int limit = (int) context.getOrDefault("limit", 10);

        LOG.info("Executing memory query: '{}' in namespace: {}", query, namespace);

        ContextConfig config = ContextConfig.builder()
                .maxMemories(limit)
                .includeMetadata(true)
                .build();

        return contextService.buildContext(query, namespace, config)
                .map(engineeredContext -> {
                    List<Map<String, Object>> results = new ArrayList<>();

                    for (ContextSection section : engineeredContext.getSections()) {
                        if (section.getType().startsWith("memory_")) {
                            results.add(Map.of(
                                    "content", section.getContent(),
                                    "tokens", section.getTokenCount(),
                                    "relevance", section.getRelevanceScore()));
                        }
                    }

                    Map<String, Object> output = Map.of(
                            "query", query,
                            "resultsCount", results.size(),
                            "results", results,
                            "contextTokens", engineeredContext.getTotalTokens());

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
}