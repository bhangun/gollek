package tech.kayys.golek.provider.openai;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * OpenAI streaming choice
 */
public class OpenAIStreamChoice {

    private int index;
    private OpenAIMessage delta;

    @JsonProperty("finish_reason")
    private String finishReason;

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public OpenAIMessage getDelta() {
        return delta;
    }

    public void setDelta(OpenAIMessage delta) {
        this.delta = delta;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public void setFinishReason(String finishReason) {
        this.finishReason = finishReason;
    }
}