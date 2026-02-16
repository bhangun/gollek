package tech.kayys.golek.spi.inference;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.auth.ApiKeyConstants;

public record BatchInferenceRequest(
                String modelId,
                @Deprecated String apiKey,
                List<Map<String, Object>> inputs,
                Map<String, Object> parameters,
                String callbackUrl) {
        public String apiKey() {
                if (apiKey == null || apiKey.isBlank()) {
                        return ApiKeyConstants.COMMUNITY_API_KEY;
                }
                return apiKey;
        }

        public List<InferenceRequest> getRequests() {
                if (inputs == null)
                        return List.of();
                return inputs.stream()
                                .map(input -> InferenceRequest.builder()
                                                .requestId(UUID.randomUUID().toString())
                                                .apiKey(apiKey())
                                                .model(modelId)
                                                .messages(Collections.<Message>emptyList())
                                                .parameters(input)
                                                .build())
                                .toList();
        }
}
