package tech.kayys.golek.core.inference;

import tech.kayys.golek.spi.inference.InferenceRequest;

/**
 * Batch inference request.
 */
public record BatchInferenceRequest(
                java.util.List<InferenceRequest> requests,
                Integer maxConcurrent) {
}