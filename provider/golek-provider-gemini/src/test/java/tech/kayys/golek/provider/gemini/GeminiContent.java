package tech.kayys.golek.provider.gemini;

import java.util.List;

public class GeminiContent {
    private List<GeminiPart> parts;
    private String role;

    public List<GeminiPart> getParts() {
        return parts;
    }

    public void setParts(List<GeminiPart> parts) {
        this.parts = parts;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }
}
