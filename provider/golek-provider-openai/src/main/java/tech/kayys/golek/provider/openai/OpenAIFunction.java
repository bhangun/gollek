package tech.kayys.golek.provider.openai;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.Map;

/**
 * OpenAI function definition
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OpenAIFunction {

    private String name;
    private String description;
    private Map<String, Object> parameters;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters;
    }
}
