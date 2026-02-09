package tech.kayys.golek.provider.mistral;

public class MistralMessage {
    private String role;
    private String content;

    public MistralMessage() {
    }

    public MistralMessage(String role, String content) {
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
