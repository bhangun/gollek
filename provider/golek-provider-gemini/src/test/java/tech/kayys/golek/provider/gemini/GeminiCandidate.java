package tech.kayys.golek.provider.gemini;

public class GeminiCandidate {
    private GeminiContent content;
    private String finishReason;

    public GeminiContent getContent() {
        return content;
    }

    public void setContent(GeminiContent content) {
        this.content = content;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public void setFinishReason(String finishReason) {
        this.finishReason = finishReason;
    }
}
