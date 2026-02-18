package tech.kayys.gollek.provider.gemini;

import java.util.List;

/**
 * Gemini API response
 */
public class GeminiResponse {

    private List<GeminiCandidate> candidates;
    private GeminiUsageMetadata usageMetadata;
    private String promptFeedback;

    public List<GeminiCandidate> getCandidates() {
        return candidates;
    }

    public void setCandidates(List<GeminiCandidate> candidates) {
        this.candidates = candidates;
    }

    public GeminiUsageMetadata getUsageMetadata() {
        return usageMetadata;
    }

    public void setUsageMetadata(GeminiUsageMetadata usageMetadata) {
        this.usageMetadata = usageMetadata;
    }

    public String getPromptFeedback() {
        return promptFeedback;
    }

    public void setPromptFeedback(String promptFeedback) {
        this.promptFeedback = promptFeedback;
    }
}
