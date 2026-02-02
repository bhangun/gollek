package tech.kayys.golek.provider.gemini;

import java.util.List;

public class GeminiResponse {
    private List<GeminiCandidate> candidates;
    private GeminiUsageMetadata usageMetadata;

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
}
