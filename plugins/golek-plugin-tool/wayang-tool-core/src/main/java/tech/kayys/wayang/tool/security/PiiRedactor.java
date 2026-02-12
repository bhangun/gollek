package tech.kayys.wayang.tool.security;

import jakarta.enterprise.context.ApplicationScoped;
import java.util.Map;

@ApplicationScoped
public class PiiRedactor {
    public Map<String, Object> redact(Map<String, Object> output, java.util.Set<String> patterns) {
        return output; // Stub implementation
    }
}
