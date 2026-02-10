package tech.kayys.wayang.vector;

import java.util.List;
import java.util.Map;

/**
 * Represents an entry in the vector store.
 */
public record VectorEntry(
        String id,
        List<Float> vector,
        String content,
        Map<String, Object> metadata) {
}
