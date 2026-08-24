package tech.kayys.gollek.server.api.v1;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stub implementation of AgentCapabilitiesMapper.
 * Returns a static capabilities document until the full client-agent library is available.
 */
public class AgentCapabilitiesMapper {

    private AgentCapabilitiesMapper() {}

    public static Map<String, Object> capabilities() {
        Map<String, Object> caps = new LinkedHashMap<>();
        caps.put("engine", "gollek");
        caps.put("role", "serving");
        caps.put("openai_compat", true);
        caps.put("grpc", true);
        caps.put("endpoints", List.of(
                "/v1/completions",
                "/v1/completions/stream",
                "/v1/gollek/chat",
                "/v1/models",
                "/q/metrics",
                "/q/health"
        ));
        caps.put("telemetry", Map.of(
                "metrics", "prometheus",
                "logs", "slf4j",
                "trace", "opentelemetry"
        ));
        return caps;
    }
}
