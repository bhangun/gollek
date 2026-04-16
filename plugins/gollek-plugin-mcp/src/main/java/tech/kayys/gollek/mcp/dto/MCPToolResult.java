package tech.kayys.gollek.mcp.dto;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Result of an MCP tool execution.
 * Compliant with MCP specification 2025-11-25.
 * 
 * <p>Tool results contain:
 * <ul>
 *   <li>content: array of content blocks (text, image, audio, resource)</li>
 *   <li>isError: true if tool execution failed (client should handle gracefully)</li>
 * </ul>
 */
public class MCPToolResult {

    private final String toolName;
    private final List<MCPContentBlock> content;
    private final Boolean isError;
    private final String errorMessage;
    private final Map<String, Object> rawOutput; // For compatibility with non-content-block based results

    private MCPToolResult(Builder builder) {
        this.toolName = builder.toolName;
        this.content = builder.content;
        this.isError = builder.isError;
        this.errorMessage = builder.errorMessage;
        this.rawOutput = builder.rawOutput;
    }

    /**
     * Name of the tool that produced this result.
     */
    public String getToolName() {
        return toolName;
    }

    /**
     * Array of content blocks.
     * Per spec: can contain text, image, audio, or resource content.
     */
    public List<MCPContentBlock> getContent() {
        return content;
    }

    /**
     * True if tool execution failed.
     * Per spec: clients MUST handle this gracefully instead of throwing JSON-RPC errors.
     */
    public Boolean getIsError() {
        return isError;
    }

    /**
     * Check if this result indicates an error.
     */
    public boolean failed() {
        return Boolean.TRUE.equals(isError);
    }

    /**
     * Check if this result indicates success.
     */
    public boolean isSuccess() {
        return !failed();
    }

    /**
     * Get the error message if tool failed.
     */
    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Extract the first text content block's text value.
     */
    public String firstText() {
        if (content == null || content.isEmpty()) return "";
        return content.stream()
                .filter(MCPContentBlock::isText)
                .map(MCPContentBlock::getText)
                .findFirst()
                .orElse("");
    }

    /**
     * Combines all text content blocks.
     */
    public String getAllText() {
        if (content == null) return "";
        return content.stream()
                .filter(c -> c.isText() && c.getText() != null)
                .map(MCPContentBlock::getText)
                .reduce("", (a, b) -> a.isEmpty() ? b : a + "\n" + b);
    }

    /**
     * Get all image content blocks.
     */
    public List<MCPContentBlock> getImages() {
        if (content == null) return List.of();
        return content.stream()
                .filter(MCPContentBlock::isImage)
                .collect(Collectors.toList());
    }

    public static Builder builder() {
        return new Builder();
    }

    @SuppressWarnings("unchecked")
    public static MCPToolResult fromMap(Map<String, Object> data) {
        if (data == null) return null;
        
        Builder builder = new Builder();
        
        // Parse toolName if present
        if (data.containsKey("toolName")) {
            builder.toolName((String) data.get("toolName"));
        }

        // Parse isError
        Object isErrorObj = data.get("isError");
        if (isErrorObj instanceof Boolean) {
            builder.isError((Boolean) isErrorObj);
        }
        
        // Parse content array
        if (data.containsKey("content")) {
            List<Map<String, Object>> contentList = (List<Map<String, Object>>) data.get("content");
            if (contentList != null) {
                builder.content(contentList.stream()
                        .map(MCPContentBlock::fromMap)
                        .collect(Collectors.toList()));
            }
        }
        
        return builder.build();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new java.util.HashMap<>();
        if (content != null) {
            map.put("content", content.stream().map(MCPContentBlock::toMap).collect(Collectors.toList()));
        }
        if (isError != null) {
            map.put("isError", isError);
        }
        return map;
    }

    public static class Builder {
        private String toolName;
        private List<MCPContentBlock> content;
        private Boolean isError;
        private String errorMessage;
        private Map<String, Object> rawOutput;

        public Builder toolName(String toolName) {
            this.toolName = toolName;
            return this;
        }

        public Builder content(List<MCPContentBlock> content) {
            this.content = content;
            return this;
        }

        public Builder isError(Boolean isError) {
            this.isError = isError;
            return this;
        }
        
        public Builder success(boolean success) {
            this.isError = !success;
            return this;
        }

        public Builder errorMessage(String message) {
            this.isError = true;
            this.errorMessage = message;
            return addText("Error: " + message);
        }

        public Builder output(Map<String, Object> output) {
            this.rawOutput = output;
            // Optionally convert to text block for visibility
            return addText(output.toString());
        }

        /**
         * Convenience method to add a text content block.
         */
        public Builder addText(String text) {
            if (this.content == null) {
                this.content = new java.util.ArrayList<>();
            }
            this.content.add(MCPContentBlock.text(text));
            return this;
        }

        /**
         * Convenience method to add an image content block.
         */
        public Builder addImage(String base64Data, String mimeType) {
            if (this.content == null) {
                this.content = new java.util.ArrayList<>();
            }
            this.content.add(MCPContentBlock.image(base64Data, mimeType));
            return this;
        }

        public MCPToolResult build() {
            return new MCPToolResult(this);
        }
    }
}