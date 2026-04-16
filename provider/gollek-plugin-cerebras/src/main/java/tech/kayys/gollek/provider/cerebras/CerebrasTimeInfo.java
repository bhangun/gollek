package tech.kayys.gollek.provider.cerebras;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Cerebras time info DTO
 */
public class CerebrasTimeInfo {

    @JsonProperty("queue_time")
    private Double queueTime;

    @JsonProperty("prompt_time")
    private Double promptTime;

    @JsonProperty("completion_time")
    private Double completionTime;

    @JsonProperty("total_time")
    private Double totalTime;

    @JsonProperty("created")
    private Long created;

    public Double getQueueTime() {
        return queueTime;
    }

    public void setQueueTime(Double queueTime) {
        this.queueTime = queueTime;
    }

    public Double getPromptTime() {
        return promptTime;
    }

    public void setPromptTime(Double promptTime) {
        this.promptTime = promptTime;
    }

    public Double getCompletionTime() {
        return completionTime;
    }

    public void setCompletionTime(Double completionTime) {
        this.completionTime = completionTime;
    }

    public Double getTotalTime() {
        return totalTime;
    }

    public void setTotalTime(Double totalTime) {
        this.totalTime = totalTime;
    }

    public Long getCreated() {
        return created;
    }

    public void setCreated(Long created) {
        this.created = created;
    }
}
