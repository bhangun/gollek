package tech.kayys.golek.provider.cerebras;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Cerebras usage DTO
 */
public class CerebrasUsage {

    @JsonProperty("prompt_tokens")
    private int promptTokens;

    @JsonProperty("completion_tokens")
    private int completionTokens;

    @JsonProperty("total_tokens")
    private int totalTokens;

    @JsonProperty("time_info")
    private CerebrasTimeInfo timeInfo;

    public int getPromptTokens() {
        return promptTokens;
    }

    public void setPromptTokens(int promptTokens) {
        this.promptTokens = promptTokens;
    }

    public int getCompletionTokens() {
        return completionTokens;
    }

    public void setCompletionTokens(int completionTokens) {
        this.completionTokens = completionTokens;
    }

    public int getTotalTokens() {
        return totalTokens;
    }

    public void setTotalTokens(int totalTokens) {
        this.totalTokens = totalTokens;
    }

    public CerebrasTimeInfo getTimeInfo() {
        return timeInfo;
    }

    public void setTimeInfo(CerebrasTimeInfo timeInfo) {
        this.timeInfo = timeInfo;
    }
}