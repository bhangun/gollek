package tech.kayys.wayang.vector.runtime;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import tech.kayys.wayang.vector.VectorStore;

/**
 * Factory and producer for VectorStore implementations.
 */
@ApplicationScoped
public class VectorStoreProvider {

    @ConfigProperty(name = "wayang.vector.store.type", defaultValue = "in-memory")
    String vectorStoreType;

    private volatile VectorStore vectorStore;

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

    private VectorStore createVectorStore(String type) {
        switch (type.toLowerCase()) {
            case "in-memory":
            case "inmemory":
                return new InMemoryVectorStore();
            case "pgvector":
                return new PgVectorStore();
            case "qdrant":
                return new QdrantVectorStore();
            case "milvus":
                return new MilvusVectorStore();
            case "chroma":
                return new ChromaVectorStore();
            case "pinecone":
                return new PineconeVectorStore();
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