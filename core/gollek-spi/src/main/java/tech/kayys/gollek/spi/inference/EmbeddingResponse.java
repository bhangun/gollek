package tech.kayys.gollek.spi.inference;

import java.util.List;
import java.util.Map;

/**
 * Response containing generated embeddings.
 */
public record EmbeddingResponse(
        String requestId,
        String model,
        List<float[]> embeddings,
        int dimension,
        Map<String, Object> metadata) {

    public EmbeddingResponse {
        if (metadata == null) {
            metadata = Map.of();
        }
    }
}
