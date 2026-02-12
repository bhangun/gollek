package tech.kayys.golek.spi.inference;

import io.smallrye.mutiny.Multi;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.auth.ApiKeyConstants;

public record StreamingSession(
        String sessionId,
        String modelId,
        @Deprecated String tenantId,
        Multi<Message> stream
) {
    public String apiKey() {
        if (tenantId == null || tenantId.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return tenantId;
    }
}
