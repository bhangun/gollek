package tech.kayys.gollek.provider.cerebras;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Cerebras choice DTO
 */
public class CerebrasChoice {

    private int index;
    private CerebrasMessage message;

    @JsonProperty("finish_reason")
    private String finishReason;

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public CerebrasMessage getMessage() {
        return message;
    }

    public void setMessage(CerebrasMessage message) {
        this.message = message;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public void setFinishReason(String finishReason) {
        this.finishReason = finishReason;
    }
}