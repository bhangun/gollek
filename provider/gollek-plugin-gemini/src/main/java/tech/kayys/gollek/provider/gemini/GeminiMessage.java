package tech.kayys.gollek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Gemini OpenAI-compatible message
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class GeminiMessage {
    private String role;
    private String content;

    public GeminiMessage() {
    }

    public GeminiMessage(String role, String content) {
        this.role = role;
        this.content = content;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }
}
