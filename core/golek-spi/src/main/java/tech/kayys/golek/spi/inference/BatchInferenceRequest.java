package tech.kayys.golek.spi.inference;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import tech.kayys.golek.spi.auth.ApiKeyConstants;

public record BatchInferenceRequest(
                String modelId,
                @Deprecated String tenantId,
                List<Map<String, Object>> inputs,
                Map<String, Object> parameters,
                String callbackUrl) {
        public String apiKey() {
                if (tenantId == null || tenantId.isBlank()) {
                        return ApiKeyConstants.COMMUNITY_API_KEY;
                }
                return tenantId;
        }

        public List<InferenceRequest> getRequests() {
                if (inputs == null)
                        return List.of();
                return inputs.stream()
                                .map(input -> new InferenceRequest(
                                                UUID.randomUUID().toString(), // RequestID
                                                apiKey(),
                                                modelId,
                                                Collections.emptyList(), // Messages
                                                input, // Parameters
                                                null, null, false, null, null, 5, false)) // added cacheBypass
                                .toList();
        }
}
