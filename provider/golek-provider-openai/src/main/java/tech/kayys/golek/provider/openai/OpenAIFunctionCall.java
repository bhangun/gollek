package tech.kayys.golek.provider.openai;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * OpenAI function call
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OpenAIFunctionCall {

    private String name;
    private String arguments;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getArguments() {
        return arguments;
    }

    public void setArguments(String arguments) {
        this.arguments = arguments;
    }
}
