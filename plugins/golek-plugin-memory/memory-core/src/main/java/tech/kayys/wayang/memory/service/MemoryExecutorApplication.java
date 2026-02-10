package tech.kayys.gamelan.executor.memory;

import io.quarkus.runtime.Quarkus;
import io.quarkus.runtime.QuarkusApplication;
import io.quarkus.runtime.annotations.QuarkusMain;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Gamelan Memory Executor - Main Application
 */
@QuarkusMain
public class MemoryExecutorApplication implements QuarkusApplication {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryExecutorApplication.class);

    public static void main(String[] args) {
        LOG.info("Starting Gamelan Memory Executor...");
        Quarkus.run(MemoryExecutorApplication.class, args);
    }

    @Override
    public int run(String... args) throws Exception {
        LOG.info("=".repeat(60));
        LOG.info("  Gamelan Memory Executor Started");
        LOG.info("  Version: 1.0.0");
        LOG.info("  Ready to process memory-aware workflow tasks");
        LOG.info("=".repeat(60));

        Quarkus.waitForExit();
        return 0;
    }
}

/**
 * Startup initializer
 */
@ApplicationScoped
class MemoryExecutorInitializer {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryExecutorInitializer.class);

    @Inject
    VectorStoreFactory vectorStoreFactory;

    @Inject
    EmbeddingServiceFactory embeddingServiceFactory;

    void onStart(@jakarta.enterprise.event.Observes io.quarkus.runtime.StartupEvent event) {
        LOG.info("Initializing Memory Executor components...");

        // Initialize vector store
        VectorMemoryStore vectorStore = vectorStoreFactory.getVectorStore();
        LOG.info("Vector store initialized: {}", vectorStore.getClass().getSimpleName());

        // Initialize embedding service
        EmbeddingService embeddingService = embeddingServiceFactory.getEmbeddingService();
        LOG.info("Embedding service initialized: {} (dimension: {})",
                embeddingService.getProvider(),
                embeddingService.getDimension());

        LOG.info("Memory Executor initialization complete");
    }
}