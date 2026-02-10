package tech.kayys.wayang.vector.runtime;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import tech.kayys.gamelan.executor.memory.VectorMemoryStore;
import tech.kayys.wayang.vector.VectorStore;
import tech.kayys.wayang.vector.runtime.adapter.VectorStoreAdapter;

/**
 * Factory and producer for VectorStore implementations.
 */
@ApplicationScoped
public class VectorStoreProvider {

    @ConfigProperty(name = "wayang.vector.store.type", defaultValue = "in-memory")
    String vectorStoreType;

    private volatile VectorStore vectorStore;
    private volatile VectorMemoryStore vectorMemoryStore;

    @Produces
    @ApplicationScoped
    public VectorStore getVectorStore() {
        if (vectorStore == null) {
            synchronized (this) {
                if (vectorStore == null) {
                    vectorStore = createVectorStore(vectorStoreType);
                }
            }
        }
        return vectorStore;
    }

    @Produces
    @ApplicationScoped
    public VectorMemoryStore getVectorMemoryStore() {
        if (vectorMemoryStore == null) {
            synchronized (this) {
                if (vectorMemoryStore == null) {
                    VectorStore store = getVectorStore();
                    vectorMemoryStore = new VectorStoreAdapter(store);
                }
            }
        }
        return vectorMemoryStore;
    }

    private VectorStore createVectorStore(String type) {
        switch (type.toLowerCase()) {
            case "in-memory":
            case "inmemory":
                return new InMemoryVectorStore();
            case "pgvector":
                return new PgVectorStore(); // Placeholder - would be implemented separately
            case "qdrant":
                return new QdrantVectorStore(); // Placeholder - would be implemented separately
            case "milvus":
                return new MilvusVectorStore(); // Placeholder - would be implemented separately
            case "chroma":
                return new ChromaVectorStore(); // Placeholder - would be implemented separately
            case "pinecone":
                return new PineconeVectorStore(); // Placeholder - would be implemented separately
            default:
                throw new IllegalArgumentException("Unknown vector store type: " + type);
        }
    }

    /**
     * Initialize the vector store (e.g., create tables, connect to database).
     */
    public Uni<Void> initialize() {
        // Initialization logic would go here
        return Uni.createFrom().voidItem();
    }
}