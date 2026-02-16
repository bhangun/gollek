package tech.kayys.golek.provider.gemini;

/**
 * Gemini candidate response
 */
public class GeminiCandidate {

    private GeminiContent content;
    private String finishReason;
    private int index;

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

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }
}
