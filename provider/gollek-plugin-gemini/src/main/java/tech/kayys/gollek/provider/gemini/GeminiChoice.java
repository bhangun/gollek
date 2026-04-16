package tech.kayys.gollek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Gemini OpenAI-compatible choice
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class GeminiChoice {

    private int index;
    private GeminiMessage message;
    private GeminiMessage delta;
    @JsonProperty("finish_reason")
    private String finishReason;

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public GeminiMessage getMessage() {
        return message;
    }

    public void setMessage(GeminiMessage message) {
        this.message = message;
    }

    public GeminiMessage getDelta() {
        return delta;
    }

    public void setDelta(GeminiMessage delta) {
        this.delta = delta;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public void setFinishReason(String finishReason) {
        this.finishReason = finishReason;
    }
}
