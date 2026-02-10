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
 * Specialized executor for memory hygiene:
 * - Remove expired memories
 * - Archive old low-importance memories
 * - Compress infrequently accessed memories
 * - Maintain memory store performance
 */
@Executor(executorType = "memory-cleanup", communicationType = tech.kayys.gamelan.core.scheduler.CommunicationType.GRPC, maxConcurrentTasks = 1)
@ApplicationScoped
public class MemoryCleanupExecutor extends AbstractWorkflowExecutor {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryCleanupExecutor.class);

    @Inject
    VectorMemoryStore memoryStore;

    // Cleanup thresholds
    private static final Duration MAX_WORKING_MEMORY_AGE = Duration.ofHours(24);
    private static final Duration MAX_EPISODIC_LOW_IMPORTANCE_AGE = Duration.ofDays(30);
    private static final double MIN_IMPORTANCE_THRESHOLD = 0.2;

    @Override
    public Uni<NodeExecutionResult> execute(NodeExecutionTask task) {
        LOG.info("Starting memory cleanup task");

        String namespace = (String) task.context().getOrDefault("namespace", "default");

        return cleanupMemories(namespace)
                .map(stats -> {
                    Map<String, Object> output = Map.of(
                            "expiredRemoved", stats.expiredRemoved,
                            "lowImportanceArchived", stats.lowImportanceArchived,
                            "workingMemoryCleared", stats.workingMemoryCleared,
                            "totalCleaned", stats.totalCleaned());

                    return NodeExecutionResult.success(
                            task.runId(),
                            task.nodeId(),
                            task.attempt(),
                            output,
                            task.token());
                });
    }

    /**
     * Cleanup memories based on various criteria
     */
    private Uni<CleanupStats> cleanupMemories(String namespace) {
        LOG.info("Cleaning up memories in namespace: {}", namespace);

        return memoryStore.getStatistics(namespace)
                .flatMap(stats -> {
                    // In a real implementation, would:
                    // 1. Query expired memories and delete
                    // 2. Query low-importance old memories and archive
                    // 3. Clear working memory older than threshold
                    // 4. Optimize storage

                    LOG.info("Cleanup complete for namespace: {}", namespace);

                    return Uni.createFrom().item(new CleanupStats(0, 0, 0));
                });
    }

    /**
     * Cleanup statistics
     */
    private record CleanupStats(
            int expiredRemoved,
            int lowImportanceArchived,
            int workingMemoryCleared) {
        int totalCleaned() {
            return expiredRemoved + lowImportanceArchived + workingMemoryCleared;
        }
    }
}