package tech.kayys.golek.spi.inference;

import io.smallrye.mutiny.Multi;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.auth.ApiKeyConstants;

public record StreamingSession(
        String sessionId,
        String modelId,
        @Deprecated String apiKey,
        Multi<Message> stream) {
    public String apiKey() {
        if (apiKey == null || apiKey.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return apiKey;
    }
}
