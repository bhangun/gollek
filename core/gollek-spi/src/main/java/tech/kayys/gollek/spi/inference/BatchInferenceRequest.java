package tech.kayys.gollek.spi.inference;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.auth.ApiKeyConstants;

public record BatchInferenceRequest(
                String modelId,
                String apiKey,
                List<Map<String, Object>> inputs,
                Map<String, Object> parameters,
                List<InferenceRequest> requests,
                Integer maxConcurrent,
                String callbackUrl) {

        public static Builder builder() {
                return new Builder();
        }

        public static class Builder {
                private String modelId;
                private String apiKey;
                private List<Map<String, Object>> inputs;
                private Map<String, Object> parameters;
                private List<InferenceRequest> requests;
                private Integer maxConcurrent;
                private String callbackUrl;

                public Builder modelId(String modelId) {
                        this.modelId = modelId;
                        return this;
                }

                public Builder apiKey(String apiKey) {
                        this.apiKey = apiKey;
                        return this;
                }

                public Builder inputs(List<Map<String, Object>> inputs) {
                        this.inputs = inputs;
                        return this;
                }

                public Builder parameters(Map<String, Object> parameters) {
                        this.parameters = parameters;
                        return this;
                }

                public Builder requests(List<InferenceRequest> requests) {
                        this.requests = requests;
                        return this;
                }

                public Builder maxConcurrent(Integer maxConcurrent) {
                        this.maxConcurrent = maxConcurrent;
                        return this;
                }

                public Builder callbackUrl(String callbackUrl) {
                        this.callbackUrl = callbackUrl;
                        return this;
                }

                public BatchInferenceRequest build() {
                        return new BatchInferenceRequest(modelId, apiKey, inputs, parameters, requests, maxConcurrent,
                                        callbackUrl);
                }
        }

        public BatchInferenceRequest(Integer maxConcurrent, String callbackUrl,
                        String apiKey, String modelId, Map<String, Object> parameters, List<Map<String, Object>> inputs,
                        List<InferenceRequest> requests) {
                this(modelId, apiKey, inputs, parameters, requests, maxConcurrent, callbackUrl);
        }

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
                                                .message(extractUserMessage(input))
                                                .parameters(input)
                                                .build())
                                .toList();
        }

        /**
         * Extracts a {@link Message.Role#USER} message from the input map.
         * Looks for a {@code "prompt"} or {@code "content"} key; falls back to
         * a generic placeholder so the {@code InferenceRequest} builder never fails
         * its at-least-one-message validation.
         */
        private static Message extractUserMessage(Map<String, Object> input) {
                Object prompt = input.get("prompt");
                if (prompt == null) {
                        prompt = input.get("content");
                }
                String text = (prompt instanceof String s && !s.isBlank()) ? s : "infer";
                return new Message(Message.Role.USER, text);
        }
}
