package tech.kayys.gollek.mcp.dto;

import java.util.List;
import java.util.Map;

/**
 * Result of an MCP inference operation.
 */
public class MCPInferenceResult {

    private final String content;
    private final Map<String, Object> metadata;
    private final int tokensUsed;

    public MCPInferenceResult(String content, Map<String, Object> metadata, int tokensUsed) {
        this.content = content;
        this.metadata = metadata;
        this.tokensUsed = tokensUsed;
    }

    public String getContent() {
        return content;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public int getTokensUsed() {
        return tokensUsed;
    }

    public static MCPInferenceResult fromMessages(List<Message> messages) {
        StringBuilder content = new StringBuilder();
        for (Message message : messages) {
            if (content.length() > 0) {
                content.append("\n\n");
            }
            content.append(message.getContent());
        }
        return new MCPInferenceResult(content.toString(), Map.of(), 0);
    }

    public static MCPInferenceResult fromPrompt(Object promptResult) {
        return new MCPInferenceResult(
                promptResult != null ? promptResult.toString() : "",
                Map.of("source", "prompt"),
                0);
    }

    public static MCPInferenceResult fromResource(String resourceContent) {
        return new MCPInferenceResult(
                resourceContent != null ? resourceContent : "",
                Map.of("source", "resource"),
                0);
    }
}