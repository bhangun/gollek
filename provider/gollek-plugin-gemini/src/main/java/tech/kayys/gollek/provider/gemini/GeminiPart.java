package tech.kayys.gollek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Gemini content part
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class GeminiPart {

    private String text;

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
}
