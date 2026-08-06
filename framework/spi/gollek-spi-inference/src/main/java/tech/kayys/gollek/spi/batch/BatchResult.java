package tech.kayys.gollek.spi.batch;

import tech.kayys.gollek.error.ErrorPayload;
import tech.kayys.gollek.spi.inference.InferenceResponse;

/**
 * Represents an individual result within a batch execution.
 */
public record BatchResult(
        String requestId,
        InferenceResponse response,
        ErrorPayload error) {

    public boolean succeeded() {
        return error == null && response != null;
    }
}
