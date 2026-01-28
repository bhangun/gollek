package tech.kayys.golek.provider.cerebras;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Cerebras message DTO
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class CerebrasMessage {

    private String role;
    private String content;

    public CerebrasMessage() {
    }

    public CerebrasMessage(String role, String content) {
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