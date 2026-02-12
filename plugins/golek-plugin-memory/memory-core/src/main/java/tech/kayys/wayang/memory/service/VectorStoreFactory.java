package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.vector.VectorStore;
import tech.kayys.wayang.vector.runtime.VectorStoreProvider;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating appropriate vector store based on configuration
 */
@ApplicationScoped
public class VectorStoreFactory {

    private static final Logger LOG = LoggerFactory.getLogger(VectorStoreFactory.class);

    @ConfigProperty(name = "gamelan.memory.store.type", defaultValue = "inmemory")
    String storeType;

    @Inject
    VectorStoreProvider vectorStoreProvider;

    /**
     * Get configured vector store
     */
    public VectorMemoryStore getVectorStore() {
        LOG.info("Using vector store type: {}", storeType);

        // Use the vector store provider from the vector module with adapter
        VectorStore vectorStore = vectorStoreProvider.getVectorStore();
        return new VectorStoreAdapter(vectorStore);
    }
}