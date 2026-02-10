package tech.kayys.wayang.memory.spi;

import io.smallrye.mutiny.Uni;
import java.util.List;

/**
 * Interface for generating vector embeddings from text.
 */
public interface EmbeddingService {

    /**
     * Generate embedding for a single text.
     *
     * @param text the input text
     * @return Uni list of floats representing the embedding
     */
    Uni<List<Float>> embed(String text);
}
