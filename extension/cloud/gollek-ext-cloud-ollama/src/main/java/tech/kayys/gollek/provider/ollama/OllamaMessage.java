package tech.kayys.gollek.provider.ollama;

/**
 * Ollama chat message
 */
public class OllamaMessage {

    private String role;
    private String content;
    private String[] images;

    public OllamaMessage() {
    }

    public OllamaMessage(String role, String content) {
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

    public String[] getImages() {
        return images;
    }

    public void setImages(String[] images) {
        this.images = images;
    }
}