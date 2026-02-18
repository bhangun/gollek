package tech.kayys.gollek.spi.inference;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.auth.ApiKeyConstants;

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
