package tech.kayys.gollek.server.api.v1;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stub implementation of AgentReadinessIssueCatalogMapper.
 * Returns a static catalog until the full client-agent library is available.
 */
public class AgentReadinessIssueCatalogMapper {

    private AgentReadinessIssueCatalogMapper() {}

    public static List<Map<String, Object>> catalog() {
        return List.of(
                issue("MODEL_NOT_FOUND", "FATAL", "No model could be resolved for the requested ID",
                        "Pull the model with: gollek pull <model-id>"),
                issue("SDK_NOT_READY", "FATAL", "The underlying SDK is not initialized",
                        "Check Quarkus startup logs for CDI errors"),
                issue("HF_UNAVAILABLE", "WARNING", "HuggingFace registry is unreachable",
                        "Check network connectivity and retry"),
                issue("MODEL_FORMAT_UNSUPPORTED", "WARNING", "The requested format is not supported on this platform",
                        "Use GGUF or SafeTensors format instead"),
                issue("INFERENCE_TIMEOUT", "WARNING", "Inference exceeded the configured timeout",
                        "Reduce max_tokens or use streaming mode")
        );
    }

    private static Map<String, Object> issue(String code, String severity, String summary, String remediation) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("code", code);
        entry.put("severity", severity);
        entry.put("summary", summary);
        entry.put("remediation", remediation);
        return entry;
    }
}
