package tech.kayys.gollek.server.api.v1;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.ws.rs.core.HttpHeaders;
import tech.kayys.gollek.sdk.core.GollekSdk;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stub implementation of AgentPreflightMapper.
 * Performs basic readiness checks without depending on the missing client-agent library.
 */
public class AgentPreflightMapper {

    private AgentPreflightMapper() {}

    /** Extract trace-related fields from the payload for the trace context. */
    public static JsonNode tracePayload(JsonNode payload) {
        return payload;
    }

    /**
     * Run a preflight check: verifies the SDK is available and the requested model (if any) is known.
     */
    public static Map<String, Object> preflight(HttpHeaders headers, JsonNode payload, AgentTraceContext trace, GollekSdk sdk) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("request_id", trace.requestId());
        report.put("engine", "gollek");

        String modelId = null;
        if (payload != null && payload.has("model")) {
            modelId = payload.get("model").asText(null);
        }

        List<Map<String, Object>> checks = new java.util.ArrayList<>();

        // Check 1: SDK availability
        Map<String, Object> sdkCheck = new LinkedHashMap<>();
        sdkCheck.put("check", "SDK_AVAILABLE");
        sdkCheck.put("status", sdk != null ? "PASS" : "FAIL");
        checks.add(sdkCheck);

        // Check 2: Model resolution
        if (modelId != null && !modelId.isBlank()) {
            Map<String, Object> modelCheck = new LinkedHashMap<>();
            modelCheck.put("check", "MODEL_FOUND");
            modelCheck.put("model_id", modelId);
            try {
                boolean found = sdk != null && sdk.getModelInfo(modelId).isPresent();
                modelCheck.put("status", found ? "PASS" : "WARN");
                modelCheck.put("message", found ? "Model resolved" : "Model not found locally; may require pull");
            } catch (Exception e) {
                modelCheck.put("status", "WARN");
                modelCheck.put("message", "Could not resolve model: " + e.getMessage());
            }
            checks.add(modelCheck);
        }

        report.put("checks", checks);
        report.put("readiness", checks.stream().allMatch(c -> !"FAIL".equals(c.get("status"))) ? "READY" : "NOT_READY");
        report.put("gollek_trace", trace.asMap());
        return report;
    }
}
