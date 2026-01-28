package tech.kayys.golek.provider.openai;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * OpenAI tool definition
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OpenAITool {

    private String type;
    private OpenAIFunction function;

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public OpenAIFunction getFunction() {
        return function;
    }

    public void setFunction(OpenAIFunction function) {
        this.function = function;
    }
}
