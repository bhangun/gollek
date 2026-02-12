package tech.kayys.wayang.examples;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Example runner to execute all examples sequentially
 */
@ApplicationScoped
public class ExampleRunner {

    private static final Logger LOG = LoggerFactory.getLogger(ExampleRunner.class);

    @Inject
    MemoryExecutorExamples examples;

    /**
     * Run all examples sequentially
     */
    public Uni<Void> runAllExamples() {
        LOG.info("Starting Memory Executor Examples...\n");

        return examples.example1_BasicMemoryOperations()
                .chain(() -> examples.example2_SemanticSearch())
                .chain(() -> examples.example3_ContextEngineering())
                .chain(() -> examples.example4_HybridSearch())
                .chain(() -> examples.example5_TemporalDecay())
                .chain(() -> examples.example6_BatchOperations())
                .chain(() -> examples.example7_MemoryStatistics())
                .invoke(() -> LOG.info("\n=== All Examples Completed Successfully ==="));
    }
}