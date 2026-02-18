package tech.kayys.gollek.spi.inference;

import tech.kayys.gollek.spi.auth.ApiKeyConstants;

public record ValidationContext(@Deprecated String apiKey, String modelId) {
    public String apiKey() {
        if (apiKey == null || apiKey.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return apiKey;
    }
}
