package tech.kayys.golek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Gemini content part
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class GeminiPart {

    private String text;
    private GeminiInlineData inlineData;
    private GeminiFunctionCall functionCall;
    private GeminiFunctionResponse functionResponse;

    public GeminiPart() {
    }

    public GeminiPart(String text) {
        this.text = text;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public GeminiInlineData getInlineData() {
        return inlineData;
    }

    public void setInlineData(GeminiInlineData inlineData) {
        this.inlineData = inlineData;
    }

    public GeminiFunctionCall getFunctionCall() {
        return functionCall;
    }

    public void setFunctionCall(GeminiFunctionCall functionCall) {
        this.functionCall = functionCall;
    }

    public GeminiFunctionResponse getFunctionResponse() {
        return functionResponse;
    }

    public void setFunctionResponse(GeminiFunctionResponse functionResponse) {
        this.functionResponse = functionResponse;
    }
}
