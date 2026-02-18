package tech.kayys.gollek.mcp.dto;

import java.util.List;
import java.util.Map;

/**
 * Message in an MCP prompt result.
 */
public class MCPPromptMessage {
    public enum Role {
        USER, SYSTEM, ASSISTANT
    }

    private final Role role;
    private final List<Content> contents;

    public MCPPromptMessage(Role role, List<Content> contents) {
        this.role = role;
        this.contents = contents;
    }

    public Role getRole() {
        return role;
    }

    public List<Content> getContents() {
        return contents;
    }

    public static class Content {
        private final String type;
        private final String text;
        private final String data; // Base64 encoded binary data
        private final String mimeType;

        public Content(String type, String text, String data, String mimeType) {
            this.type = type;
            this.text = text;
            this.data = data;
            this.mimeType = mimeType;
        }

        public String getType() {
            return type;
        }

        public String getText() {
            return text;
        }

        public String getData() {
            return data;
        }

        public String getMimeType() {
            return mimeType;
        }
    }
}