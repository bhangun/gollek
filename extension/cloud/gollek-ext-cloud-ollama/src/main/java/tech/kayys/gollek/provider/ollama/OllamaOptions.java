package tech.kayys.gollek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Ollama generation options
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OllamaOptions {

    private Double temperature;

    @JsonProperty("top_p")
    private Double topP;

    @JsonProperty("top_k")
    private Integer topK;

    @JsonProperty("num_predict")
    private Integer numPredict;

    @JsonProperty("num_ctx")
    private Integer numCtx;

    @JsonProperty("num_gpu")
    private Integer numGpu;

    @JsonProperty("repeat_penalty")
    private Double repeatPenalty;

    private Integer seed;
    private String[] stop;

    public Double getTemperature() {
        return temperature;
    }

    public void setTemperature(Double temperature) {
        this.temperature = temperature;
    }

    public Double getTopP() {
        return topP;
    }

    public void setTopP(Double topP) {
        this.topP = topP;
    }

    public Integer getTopK() {
        return topK;
    }

    public void setTopK(Integer topK) {
        this.topK = topK;
    }

    public Integer getNumPredict() {
        return numPredict;
    }

    public void setNumPredict(Integer numPredict) {
        this.numPredict = numPredict;
    }

    public Integer getNumCtx() {
        return numCtx;
    }

    public void setNumCtx(Integer numCtx) {
        this.numCtx = numCtx;
    }

    public Integer getNumGpu() {
        return numGpu;
    }

    public void setNumGpu(Integer numGpu) {
        this.numGpu = numGpu;
    }

    public Double getRepeatPenalty() {
        return repeatPenalty;
    }

    public void setRepeatPenalty(Double repeatPenalty) {
        this.repeatPenalty = repeatPenalty;
    }

    public Integer getSeed() {
        return seed;
    }

    public void setSeed(Integer seed) {
        this.seed = seed;
    }

    public String[] getStop() {
        return stop;
    }

    public void setStop(String[] stop) {
        this.stop = stop;
    }
}