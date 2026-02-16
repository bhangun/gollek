package tech.kayys.golek.spi.inference;

import tech.kayys.golek.spi.auth.ApiKeyConstants;

public record ValidationContext(@Deprecated String apiKey, String modelId) {
    public String apiKey() {
        if (apiKey == null || apiKey.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return apiKey;
    }
}
