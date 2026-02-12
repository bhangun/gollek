package tech.kayys.wayang;

import io.quarkus.runtime.Quarkus;
import io.quarkus.runtime.QuarkusApplication;
import io.quarkus.runtime.annotations.QuarkusMain;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.vector.VectorStore;
import tech.kayys.wayang.vector.runtime.VectorStoreProvider;

/**
 * Main entry point for the Vector Store Runtime module.
 */
@QuarkusMain
public class VectorRuntimeMain {
    public static void main(String... args) {
        Quarkus.run(VectorRuntimeApp.class, args);
    }
}

@ApplicationScoped
class VectorRuntimeApp implements QuarkusApplication {

    private static final Logger LOG = LoggerFactory.getLogger(VectorRuntimeApp.class);

    @Inject
    VectorStore vectorStore;

    @Inject
    VectorStoreProvider vectorStoreProvider;

    @Override
    public int run(String... args) throws Exception {
        LOG.info("Starting Vector Store Runtime...");

        // Initialize the vector store
        vectorStoreProvider.initialize()
                .await().indefinitely();

        LOG.info("Vector Store Runtime initialized with: {}", vectorStore.getClass().getSimpleName());

        // Keep the application running
        Thread.currentThread().join();

        return 0;
    }
}
