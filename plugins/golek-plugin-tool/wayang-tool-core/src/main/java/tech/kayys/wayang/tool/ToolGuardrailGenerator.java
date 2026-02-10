package tech.kayys.wayang.tool;

import java.util.Map;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.entity.ToolGuardrails;

/**
 * Tool guardrail generator
 */
@ApplicationScoped
public class ToolGuardrailGenerator {

    public ToolGuardrails generate(
            io.swagger.v3.oas.models.PathItem.HttpMethod method,
            io.swagger.v3.oas.models.Operation operation,
            Map<String, Object> config) {

        ToolGuardrails guardrails = new ToolGuardrails();

        // Set defaults
        guardrails.setValidateInputSchema(true);
        guardrails.setValidateOutputSchema(true);
        guardrails.setSanitizeInput(true);
        guardrails.setAllowRedirects(false);
        guardrails.setLogRequests(true);

        // Apply config overrides
        if (config != null) {
            if (config.containsKey("rateLimitPerMinute")) {
                guardrails.setRateLimitPerMinute(
                        (Integer) config.get("rateLimitPerMinute"));
            }
            if (config.containsKey("maxExecutionTimeMs")) {
                guardrails.setMaxExecutionTimeMs(
                        (Integer) config.get("maxExecutionTimeMs"));
            }
        }

        return guardrails;
    }
}