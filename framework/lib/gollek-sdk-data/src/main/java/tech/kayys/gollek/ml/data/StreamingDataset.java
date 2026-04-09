package tech.kayys.gollek.ml.data;

import java.util.Iterator;
import java.util.stream.Stream;

/**
 * Represents a dataset that streams its samples instead of loading them into memory.
 * <p>
 * This is particularly useful for very large datasets (e.g., audio/video corpus
 * or massive text collections) that cannot fit into memory.
 *
 * @param <T> the type of elements in this dataset
 */
public interface StreamingDataset<T> extends Iterable<T> {

    /**
     * Returns an iterator that streams through the dataset samples.
     * Note: Depending on the implementation, multiple overlapping iterations
     * might not be supported.
     */
    @Override
    Iterator<T> iterator();

    /**
     * Returns a Java 8 Stream over the dataset.
     */
    default Stream<T> stream() {
        return java.util.stream.StreamSupport.stream(spliterator(), false);
    }
}
