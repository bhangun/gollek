package tech.kayys.gollek.spi.embedding;

import io.smallrye.mutiny.Uni;
import tech.kayys.alkhawarizm.spi.model.MultimodalContent;
import java.util.List;

/**
 * Service for generating embeddings from multimodal content.
 */
public interface EmbeddingService {

    /**
     * Generate embedding for a single content item.
     */
    Uni<EmbeddingResult> embed(MultimodalContent content, String model);

    /**
     * Generate embeddings for a batch of content items.
     */
    Uni<List<EmbeddingResult>> embedBatch(List<MultimodalContent> contents, String model);

    /**
     * Result of an embedding operation.
     */
    interface EmbeddingResult {
        float[] vector();
        boolean hasError();
        String errorMessage();
        
        default double cosineSimilarity(EmbeddingResult other) {
            float[] a = this.vector();
            float[] b = other.vector();
            if (a == null || b == null || a.length != b.length) return 0.0;
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < a.length; i++) {
                dot += (double) a[i] * b[i];
                na  += (double) a[i] * a[i];
                nb  += (double) b[i] * b[i];
            }
            return dot / (Math.sqrt(na) * Math.sqrt(nb));
        }
    }
}
