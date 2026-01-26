package tech.kayys.golek.provider.core.mcp;

import java.util.*;

/**
 * Result of MCP inference processing.
 */
public final class MCPInferenceResult {

    private final String content;
    private final Map<String, Object> metadata;
    private final int tokensUsed;

    public MCPInferenceResult(
            String content,
            Map<String, Object> metadata,
            int tokensUsed) {
        this.content = Objects.requireNonNull(content, "content");
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
        this.tokensUsed = tokensUsed;
    }

    // Getters
    public String getContent() {
        return content;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public int getTokensUsed() {
        return tokensUsed;
    }

    /**
     * Create from prompt result
     */
    public static MCPInferenceResult fromPrompt(MCPPromptResult promptResult) {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("type", "prompt");
        metadata.put("promptName", promptResult.getPromptName());
        metadata.put("messageCount", promptResult.getMessages().size());

        return new MCPInferenceResult(
                promptResult.getCombinedText(),
                metadata,
                estimateTokens(promptResult.getCombinedText()));
    }

    /**
     * Create from resource content
     */
    public static MCPInferenceResult fromResource(String resourceContent) {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("type", "resource");

        return new MCPInferenceResult(
                resourceContent,
                metadata,
                estimateTokens(resourceContent));
    }

    /**
     * Create from messages
     */
    public static MCPInferenceResult fromMessages(List<Message> messages) {
        String combined = messages.stream()
                .map(Message::getContent)
                .reduce("", (a, b) -> a + "\n" + b)
                .trim();

        Map<String, Object> metadata = new HashMap<>();
        metadata.put("type", "messages");
        metadata.put("messageCount", messages.size());

        return new MCPInferenceResult(
                combined,
                metadata,
                estimateTokens(combined));
    }

    /**
     * Simple token estimation (4 chars ≈ 1 token)
     */
    private static int estimateTokens(String text) {
        return text.length() / 4;
    }

    @Override
    public String toString() {
        return "MCPInferenceResult{" +
                "contentLength=" + content.length() +
                ", tokensUsed=" + tokensUsed +
                ", metadataKeys=" + metadata.keySet() +
                '}';
    }
}