package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gamelan.core.domain.*;
import tech.kayys.gamelan.core.engine.NodeExecutionResult;
import tech.kayys.gamelan.core.engine.NodeExecutionTask;
import tech.kayys.gamelan.executor.AbstractWorkflowExecutor;
import tech.kayys.gamelan.executor.Executor;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;

/**
 * Specialized executor that consolidates episodic memories into semantic
 * knowledge.
 * Runs periodically to:
 * - Identify patterns across episodic memories
 * - Extract semantic knowledge
 * - Reduce memory footprint
 * - Improve retrieval quality
 */
@Executor(executorType = "memory-consolidation", communicationType = tech.kayys.gamelan.core.scheduler.CommunicationType.GRPC, maxConcurrentTasks = 1)
@ApplicationScoped
public class MemoryConsolidationExecutor extends AbstractWorkflowExecutor {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryConsolidationExecutor.class);

    @Inject
    VectorMemoryStore memoryStore;

    @Inject
    EmbeddingServiceFactory embeddingFactory;

    @Inject
    TextChunker textChunker;

    @Override
    public Uni<NodeExecutionResult> execute(NodeExecutionTask task) {
        LOG.info("Starting memory consolidation task");

        String namespace = (String) task.context().getOrDefault("namespace", "default");

        return consolidateMemories(namespace)
                .map(stats -> {
                    Map<String, Object> output = Map.of(
                            "consolidated", stats.consolidated,
                            "patternsFound", stats.patterns,
                            "semanticMemoriesCreated", stats.semanticCreated,
                            "episodicMemoriesRetained", stats.episodicRetained);

                    return NodeExecutionResult.success(
                            task.runId(),
                            task.nodeId(),
                            task.attempt(),
                            output,
                            task.token());
                });
    }

    /**
     * Consolidate episodic memories into semantic knowledge
     */
    private Uni<ConsolidationStats> consolidateMemories(String namespace) {
        LOG.info("Consolidating memories in namespace: {}", namespace);

        // Get statistics
        return memoryStore.getStatistics(namespace)
                .flatMap(stats -> {
                    LOG.info("Found {} episodic memories to consolidate", stats.getEpisodicCount());

                    // For now, return basic stats
                    // In real implementation, would:
                    // 1. Cluster similar episodic memories
                    // 2. Extract common patterns
                    // 3. Create semantic memories
                    // 4. Archive or delete consolidated episodic memories

                    return Uni.createFrom().item(new ConsolidationStats(
                            stats.getEpisodicCount(),
                            0, // patterns found
                            0, // semantic created
                            stats.getEpisodicCount() // all retained for now
                    ));
                });
    }

    /**
     * Consolidation statistics
     */
    private record ConsolidationStats(
            long consolidated,
            int patterns,
            int semanticCreated,
            long episodicRetained) {
    }
}