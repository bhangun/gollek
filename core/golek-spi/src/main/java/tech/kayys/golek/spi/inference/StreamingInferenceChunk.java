package tech.kayys.golek.spi.inference;

/**
 * Streaming inference chunk.
 */
public record StreamingInferenceChunk(
                String requestId,
                int sequenceNumber,
                String token,
                boolean isComplete,
                Long latencyMs) {
}