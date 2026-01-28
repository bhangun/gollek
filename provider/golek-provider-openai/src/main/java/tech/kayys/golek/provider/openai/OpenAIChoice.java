package tech.kayys.golek.provider.openai;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * OpenAI choice
 */
public class OpenAIChoice {

    private int index;
    private OpenAIMessage message;

    @JsonProperty("finish_reason")
    private String finishReason;

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public OpenAIMessage getMessage() {
        return message;
    }

    public void setMessage(OpenAIMessage message) {
        this.message = message;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public void setFinishReason(String finishReason) {
        this.finishReason = finishReason;
    }
}