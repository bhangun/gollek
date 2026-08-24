package tech.kayys.gollek.server.api.v1;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stub implementation of AgentContractMapper.
 * Returns a static agent contract document until the full client-agent library is available.
 */
public class AgentContractMapper {

    private AgentContractMapper() {}

    public static Map<String, Object> contract() {
        Map<String, Object> contract = new LinkedHashMap<>();
        contract.put("version", "1.0");
        contract.put("engine", "gollek");
        contract.put("boundary", "serving");
        contract.put("endpoints", Map.of(
                "completions", "/v1/completions",
                "stream", "/v1/completions/stream",
                "chat", "/v1/gollek/chat",
                "models", "/v1/models",
                "health", "/q/health",
                "metrics", "/q/metrics"
        ));
        contract.put("auth", List.of("X-API-Key"));
        contract.put("formats", List.of("gguf", "safetensors", "onnx", "litert"));
        return contract;
    }
}
