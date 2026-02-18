package tech.kayys.gollek.core.inference;

import tech.kayys.gollek.spi.inference.InferenceRequest;

/**
 * Batch inference request.
 */
public record BatchInferenceRequest(
                java.util.List<InferenceRequest> requests,
                Integer maxConcurrent) {
}