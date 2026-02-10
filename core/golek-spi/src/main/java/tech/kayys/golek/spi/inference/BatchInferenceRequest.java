package tech.kayys.golek.spi.inference;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public record BatchInferenceRequest(
                String modelId,
                @Deprecated String tenantId,
                List<Map<String, Object>> inputs,
                Map<String, Object> parameters,
                String callbackUrl) {
        public List<InferenceRequest> getRequests() {
                if (inputs == null)
                        return List.of();
                return inputs.stream()
                                .map(input -> new InferenceRequest(
                                                UUID.randomUUID().toString(), // RequestID
                                                tenantId,
                                                modelId,
                                                Collections.emptyList(), // Messages
                                                input, // Parameters
                                                null, null, false, null, null, 5, false)) // added cacheBypass
                                .toList();
        }
}
