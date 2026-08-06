package tech.kayys.gollek.spi.processor;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.alkhawarizm.spi.model.MultimodalRequest;
import tech.kayys.alkhawarizm.spi.model.MultimodalResponse;

/**
 * Processor for multimodal inference.
 * Handles preprocessing, inference, and postprocessing.
 */
public interface MultimodalProcessor {

    /**
     * Process a multimodal request and return a response.
     *
     * @param request the multimodal request
     * @return Uni containing the response
     */
    Uni<MultimodalResponse> process(MultimodalRequest request);

    /**
     * Process a multimodal request with streaming response.
     *
     * @param request the multimodal request
     * @return Multi containing streaming chunks
     */
    Multi<MultimodalResponse> processStream(MultimodalRequest request);

    /**
     * Get the processor ID.
     *
     * @return processor ID
     */
    String getProcessorId();

    /**
     * Check if the processor is available.
     *
     * @return true if available
     */
    boolean isAvailable();
}
