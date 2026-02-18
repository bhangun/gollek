package tech.kayys.gollek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Ollama streaming chunk
 */
public class OllamaStreamChunk {

    private String model;
    private String createdAt;
    private OllamaMessage message;
    private boolean done;

    @JsonProperty("eval_count")
    private int evalCount;

    @JsonProperty("eval_duration")
    private long evalDuration;

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public String getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(String createdAt) {
        this.createdAt = createdAt;
    }

    public OllamaMessage getMessage() {
        return message;
    }

    public void setMessage(OllamaMessage message) {
        this.message = message;
    }

    public boolean isDone() {
        return done;
    }

    public void setDone(boolean done) {
        this.done = done;
    }

    public int getEvalCount() {
        return evalCount;
    }

    public void setEvalCount(int evalCount) {
        this.evalCount = evalCount;
    }

    public long getEvalDuration() {
        return evalDuration;
    }

    public void setEvalDuration(long evalDuration) {
        this.evalDuration = evalDuration;
    }
}
