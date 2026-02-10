package tech.kayys.golek.spi.inference;

import io.smallrye.mutiny.Multi;
import tech.kayys.golek.spi.Message;

public record StreamingSession(
        String sessionId,
        String modelId,
        String tenantId,
        Multi<Message> stream
) {
}