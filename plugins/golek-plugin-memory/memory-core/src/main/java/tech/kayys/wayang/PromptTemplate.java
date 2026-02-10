package tech.kayys.gamelan.executor.memory;

import java.util.List;
import java.util.Map;

/**
 * Prompt template
 */
public class PromptTemplate {

    private final String name;
    private final String template;
    private final List<String> requiredVariables;

    public PromptTemplate(String name, String template, List<String> requiredVariables) {
        this.name = name;
        this.template = template;
        this.requiredVariables = requiredVariables;
    }

    /**
     * Apply template with variables
     */
    public String apply(Map<String, String> variables) {
        String result = template;

        // Validate required variables
        for (String required : requiredVariables) {
            if (!variables.containsKey(required)) {
                throw new IllegalArgumentException("Missing required variable: " + required);
            }
        }

        // Replace variables
        for (Map.Entry<String, String> entry : variables.entrySet()) {
            String placeholder = "{" + entry.getKey() + "}";
            result = result.replace(placeholder, entry.getValue());
        }

        return result;
    }

    public String getName() {
        return name;
    }

    public String getTemplate() {
        return template;
    }

    public List<String> getRequiredVariables() {
        return requiredVariables;
    }
}