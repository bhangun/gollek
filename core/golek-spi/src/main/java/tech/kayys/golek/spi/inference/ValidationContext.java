package tech.kayys.golek.spi.inference;

import tech.kayys.golek.spi.auth.ApiKeyConstants;

public record ValidationContext(@Deprecated String tenantId, String modelId) {
    public String apiKey() {
        if (tenantId == null || tenantId.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return tenantId;
    }
}
