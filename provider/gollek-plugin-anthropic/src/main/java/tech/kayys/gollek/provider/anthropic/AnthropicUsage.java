package tech.kayys.gollek.provider.anthropic;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Anthropic token usage.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class AnthropicUsage {

    @JsonProperty("input_tokens")
    private Integer inputTokens;

    @JsonProperty("output_tokens")
    private Integer outputTokens;

    public AnthropicUsage() {
    }

    public Integer getInputTokens() {
        return inputTokens;
    }

    public void setInputTokens(Integer inputTokens) {
        this.inputTokens = inputTokens;
    }

    public Integer getOutputTokens() {
        return outputTokens;
    }

    public void setOutputTokens(Integer outputTokens) {
        this.outputTokens = outputTokens;
    }
}
