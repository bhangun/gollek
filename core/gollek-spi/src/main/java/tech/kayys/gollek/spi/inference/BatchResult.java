package tech.kayys.gollek.spi.inference;

import tech.kayys.gollek.spi.error.ErrorPayload;

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
