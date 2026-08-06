package tech.kayys.gollek.spi.inference;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.auth.ApiKeyConstants;
import tech.kayys.gollek.spi.context.RequestContext;

public record StreamingSession(
        String sessionId,
        String modelId,
        RequestContext requestContext,
        Multi<Message> stream) {
    public String apiKey() {
        if (requestContext.apiKey() == null || requestContext.apiKey().isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return requestContext.apiKey();
    }
}
