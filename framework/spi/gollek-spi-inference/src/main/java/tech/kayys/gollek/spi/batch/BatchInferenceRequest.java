package tech.kayys.gollek.spi.batch;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;

public record BatchInferenceRequest(
                String requestId,
                String modelId,
                tech.kayys.gollek.spi.context.RequestContext requestContext,
                List<Map<String, Object>> inputs,
                Map<String, Object> parameters,
                List<InferenceRequest> requests,
                Integer maxConcurrent,
                String callbackUrl) {

        public static Builder builder() {
                return new Builder();
        }

        public static class Builder {
                private String requestId;
                private String modelId;
                private tech.kayys.gollek.spi.context.RequestContext requestContext;
                private List<Map<String, Object>> inputs;
                private Map<String, Object> parameters;
                private List<InferenceRequest> requests;
                private Integer maxConcurrent;
                private String callbackUrl;

                public Builder requestContext(tech.kayys.gollek.spi.context.RequestContext requestContext) {
                        this.requestContext = requestContext;
                        return this;
                }

                public Builder apiKey(String apiKey) {
                        this.requestContext = tech.kayys.gollek.spi.context.RequestContext
                                        .of(requestId != null ? requestId : UUID.randomUUID().toString());
                        // Note: Simple RequestContext.of only takes requestId.
                        // For better multitenancy, we should probably have a better way to set apiKey
                        // in Builder.
                        return this;
                }

                public Builder requestId(String requestId) {
                        this.requestId = requestId;
                        return this;
                }

                public Builder modelId(String modelId) {
                        this.modelId = modelId;
                        return this;
                }

                public Builder apiKey(String apiKey, String requestId) {
                        this.requestContext = new tech.kayys.gollek.spi.context.RequestContext() {
                                @Override
                                public String apiKey() {
                                        return apiKey;
                                }

                                @Override
                                public String getRequestId() {
                                        return requestId;
                                }
                        };
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
                        if (requestContext == null) {
                                requestContext = tech.kayys.gollek.spi.context.RequestContext.of(requestId);
                        }
                        return new BatchInferenceRequest(requestId, modelId, requestContext, inputs, parameters,
                                        requests,
                                        maxConcurrent,
                                        callbackUrl);
                }
        }

        public BatchInferenceRequest(Integer maxConcurrent, String callbackUrl,
                        String apiKey, String modelId, Map<String, Object> parameters, List<Map<String, Object>> inputs,
                        List<InferenceRequest> requests) {
                this(UUID.randomUUID().toString(), modelId,
                                tech.kayys.gollek.spi.context.RequestContext.of(UUID.randomUUID().toString()), // simplistic
                                inputs, parameters, requests, maxConcurrent,
                                callbackUrl);
        }

        public String apiKey() {
                return requestContext != null ? requestContext.apiKey()
                                : tech.kayys.gollek.spi.context.RequestContext.COMMUNITY_API_KEY;
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
