package tech.kayys.golek.provider.gemini;

import java.util.List;

/**
 * Gemini candidate response
 */
public class GeminiCandidate {

    private GeminiContent content;
    private String finishReason;
    private int index;
    private List<GeminiSafetyRating> safetyRatings;

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

    public List<GeminiSafetyRating> getSafetyRatings() {
        return safetyRatings;
    }

    public void setSafetyRatings(List<GeminiSafetyRating> safetyRatings) {
        this.safetyRatings = safetyRatings;
    }
}
