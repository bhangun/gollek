package tech.kayys.golek.provider.gemini;

public class GeminiPart {
    private String text;

    public GeminiPart(String text) {
        this.text = text;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }
}
