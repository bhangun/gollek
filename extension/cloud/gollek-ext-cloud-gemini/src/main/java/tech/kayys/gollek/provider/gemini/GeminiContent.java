package tech.kayys.gollek.provider.gemini;

import java.util.List;

/**
 * Gemini content (message)
 */
public class GeminiContent {

    private String role;
    private List<GeminiPart> parts;

    public GeminiContent() {
    }

    public GeminiContent(String role, List<GeminiPart> parts) {
        this.role = role;
        this.parts = parts;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    public List<GeminiPart> getParts() {
        return parts;
    }

    public void setParts(List<GeminiPart> parts) {
        this.parts = parts;
    }
}
