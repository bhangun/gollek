package tech.kayys.golek.provider.openai;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * OpenAI tool call
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OpenAIToolCall {

    private String id;
    private String type;
    private OpenAIFunctionCall function;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public OpenAIFunctionCall getFunction() {
        return function;
    }

    public void setFunction(OpenAIFunctionCall function) {
        this.function = function;
    }
}
