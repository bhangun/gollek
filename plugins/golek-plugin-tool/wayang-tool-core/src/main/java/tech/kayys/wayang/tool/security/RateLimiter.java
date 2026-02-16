package tech.kayys.wayang.tool.security;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.entity.ToolGuardrails;

@ApplicationScoped
public class RateLimiter {
    public void checkLimit(String requestId, String toolId, ToolGuardrails guardrails) {
        // Stub implementation
    }
}
