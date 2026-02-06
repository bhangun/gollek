package tech.kayys.golek.engine.resource.dto;

import tech.kayys.golek.spi.inference.InferenceResponse;
import java.time.Instant;

public record InferenceResponseDTO(
        String requestId,
        String content,
        String model,
        int tokensUsed,
        long durationMs,
        Instant timestamp) {
    public static InferenceResponseDTO from(InferenceResponse response) {
        return new InferenceResponseDTO(
                response.getRequestId(),
                response.getContent(),
                response.getModel(),
                response.getTokensUsed(),
                response.getDurationMs(),
                response.getTimestamp());
    }
}
