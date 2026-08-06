package tech.kayys.gollek.spi.inference;

import tech.kayys.gollek.spi.auth.ApiKeyConstants;
import tech.kayys.gollek.spi.context.RequestContext;

public record ValidationContext(RequestContext requestContext, String modelId) {
    public String apiKey() {
        if (requestContext == null || requestContext.getApiKey().isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return requestContext.getApiKey();
    }

    public String requestId() {
        if (requestContext == null) {
            return null;
        }
        return requestContext.getRequestId();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private RequestContext requestContext;
        private String modelId;

        public Builder requestContext(RequestContext requestContext) {
            this.requestContext = requestContext;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public ValidationContext build() {
            return new ValidationContext(requestContext, modelId);
        }
    }
}
