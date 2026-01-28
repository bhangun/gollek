package tech.kayys.golek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonProperty;

public class OllamaPullProgress {
    @JsonProperty("status")
    private String status;

    @JsonProperty("digest")
    private String digest;

    @JsonProperty("total")
    private Long total;

    @JsonProperty("completed")
    private Long completed;

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getDigest() {
        return digest;
    }

    public void setDigest(String digest) {
        this.digest = digest;
    }

    public Long getTotal() {
        return total;
    }

    public void setTotal(Long total) {
        this.total = total;
    }

    public Long getCompleted() {
        return completed;
    }

    public void setCompleted(Long completed) {
        this.completed = completed;
    }
}
