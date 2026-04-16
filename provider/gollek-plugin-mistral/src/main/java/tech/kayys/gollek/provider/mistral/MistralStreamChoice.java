package tech.kayys.gollek.provider.mistral;

import com.fasterxml.jackson.annotation.JsonProperty;

public class MistralStreamChoice {
    private int index;
    private MistralDelta delta;
    @JsonProperty("finish_reason")
    private String finishReason;

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public MistralDelta getDelta() {
        return delta;
    }

    public void setDelta(MistralDelta delta) {
        this.delta = delta;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public void setFinishReason(String finishReason) {
        this.finishReason = finishReason;
    }
}
