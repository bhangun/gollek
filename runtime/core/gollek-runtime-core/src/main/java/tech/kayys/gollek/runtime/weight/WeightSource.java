package tech.kayys.gollek.runtime.weight;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.Set;

/**
 * Represents a source of weights, such as a Safetensors file or a directory.
 */
public interface WeightSource extends AutoCloseable {
    /**
     * Retrieves a tensor by its key.
     */
    Tensor get(String key);

    /**
     * Checks if the source contains a weight with the given key.
     */
    boolean contains(String key);

    /**
     * Returns the set of all available keys in this source.
     */
    Set<String> keys();

    @Override
    default void close() throws Exception {
        // Optional cleanup
    }
}
