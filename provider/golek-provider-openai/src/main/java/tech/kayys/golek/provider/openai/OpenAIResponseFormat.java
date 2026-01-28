package tech.kayys.golek.provider.openai;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * OpenAI response format
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OpenAIResponseFormat {

    private String type;

    public OpenAIResponseFormat() {
    }

    public OpenAIResponseFormat(String type) {
        this.type = type;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }
}
