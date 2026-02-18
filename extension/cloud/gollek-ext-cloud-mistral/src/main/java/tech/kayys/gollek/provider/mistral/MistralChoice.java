package tech.kayys.gollek.provider.mistral;

import com.fasterxml.jackson.annotation.JsonProperty;

public class MistralChoice {
    private int index;
    private MistralMessage message;
    @JsonProperty("finish_reason")
    private String finishReason;

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public MistralMessage getMessage() {
        return message;
    }

    public void setMessage(MistralMessage message) {
        this.message = message;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public void setFinishReason(String finishReason) {
        this.finishReason = finishReason;
    }
}
