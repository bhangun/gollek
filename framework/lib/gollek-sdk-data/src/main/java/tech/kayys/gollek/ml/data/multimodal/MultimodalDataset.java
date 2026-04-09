package tech.kayys.gollek.ml.data.multimodal;

import tech.kayys.gollek.ml.data.Dataset;
import tech.kayys.gollek.spi.model.MultimodalContent;

import java.util.List;

/**
 * A generic multimodal dataset that yields lists of {@link MultimodalContent} objects.
 *
 * <p>This dataset implementation provides flexibility for handling collections of multimodal data,
 * where each sample contains a list of content items that may include mixed modalities (e.g., text,
 * images, audio, video, or custom modalities).
 *
 * <p><b>Use Cases:</b>
 * <ul>
 *   <li>Pre-loaded multimodal datasets: Data generated dynamically or loaded from JSONL format</li>
 *   <li>Mixed modality collections: Handle variable numbers of modalities per sample</li>
 *   <li>Flexible data sources: Data can be sourced from any origin (database, API, file, etc.)</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>
 * List&lt;List&lt;MultimodalContent&gt;&gt; samples = List.of(
 *     List.of(
 *         MultimodalContent.text("A cat"),
 *         MultimodalContent.image(imageTensor1),
 *         MultimodalContent.audio(audioBytes1)
 *     ),
 *     List.of(
 *         MultimodalContent.text("A dog"),
 *         MultimodalContent.image(imageTensor2),
 *         MultimodalContent.audio(audioBytes2)
 *     )
 * );
 * 
 * MultimodalDataset dataset = new MultimodalDataset(samples);
 * List&lt;MultimodalContent&gt; firstSample = dataset.get(0);
 * </pre>
 *
 * <p><b>Thread Safety:</b> This class is thread-safe for reading. Modifications to the backing
 * store must be handled externally.
 *
 * @see MultimodalContent
 * @see Dataset
 */
public class MultimodalDataset implements Dataset<List<MultimodalContent>> {

    private final List<List<MultimodalContent>> backingStore;

    /**
     * Constructs a multimodal dataset with pre-loaded data.
     *
     * @param preloadedData a list of multimodal content lists, where each inner list
     *                      represents a single sample containing one or more content items
     *                      of potentially different modalities. Must not be null.
     * @throws NullPointerException if {@code preloadedData} is null
     */
    public MultimodalDataset(List<List<MultimodalContent>> preloadedData) {
        this.backingStore = preloadedData;
    }

    /**
     * Retrieves a sample at the specified index.
     *
     * <p>Each sample is a list of {@link MultimodalContent} objects representing different
     * modalities (text, image, audio, etc.) for that sample.
     *
     * @param index the zero-based index of the sample to retrieve
     * @return a list of multimodal content for the requested sample
     * @throws IndexOutOfBoundsException if the index is out of range
     *         ({@code index < 0 || index >= size()})
     */
    @Override
    public List<MultimodalContent> get(int index) {
        return backingStore.get(index);
    }

    /**
     * Returns the total number of samples in this dataset.
     *
     * @return the number of multimodal samples available in this dataset
     */
    @Override
    public int size() {
        return backingStore.size();
    }
}
