package tech.kayys.golek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Gemini generation config
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class GeminiGenerationConfig {

    private Double temperature;
    private Double topP;
    private Integer topK;
    private Integer candidateCount;
    private Integer maxOutputTokens;
    private String[] stopSequences;
    private String responseMimeType;

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

    public Integer getCandidateCount() {
        return candidateCount;
    }

    public void setCandidateCount(Integer candidateCount) {
        this.candidateCount = candidateCount;
    }

    public Integer getMaxOutputTokens() {
        return maxOutputTokens;
    }

    public void setMaxOutputTokens(Integer maxOutputTokens) {
        this.maxOutputTokens = maxOutputTokens;
    }

    public String[] getStopSequences() {
        return stopSequences;
    }

    public void setStopSequences(String[] stopSequences) {
        this.stopSequences = stopSequences;
    }

    public String getResponseMimeType() {
        return responseMimeType;
    }

    public void setResponseMimeType(String responseMimeType) {
        this.responseMimeType = responseMimeType;
    }
}
