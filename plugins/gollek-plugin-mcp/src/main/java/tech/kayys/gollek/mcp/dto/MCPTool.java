package tech.kayys.gollek.mcp.dto;

import java.util.Map;

/**
 * Represents an MCP tool definition.
 * Compliant with MCP specification 2025-11-25.
 * 
 * <p>Tool definitions include:
 * <ul>
 *   <li>name: unique tool identifier</li>
 *   <li>description: human-readable description</li>
 *   <li>inputSchema: JSON Schema 2020-12 for tool arguments</li>
 *   <li>annotations: optional hints about tool behavior</li>
 * </ul>
 */
public class MCPTool {

    private final String name;
    private final String description;
    private final Map<String, Object> inputSchema;
    private final ToolAnnotations annotations;

    private MCPTool(Builder builder) {
        this.name = builder.name;
        this.description = builder.description;
        this.inputSchema = builder.inputSchema;
        this.annotations = builder.annotations;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    /**
     * JSON Schema 2020-12 object defining the expected tool arguments.
     * Per spec: type MUST be "object".
     */
    public Map<String, Object> getInputSchema() {
        return inputSchema;
    }

    /**
     * Optional hints about tool behavior.
     * Per spec: MUST be treated as untrusted unless from verified server.
     */
    public ToolAnnotations getAnnotations() {
        return annotations;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static MCPTool fromMap(Map<String, Object> data) {
        Builder builder = new Builder()
                .name((String) data.get("name"))
                .description((String) data.get("description"))
                .inputSchema((Map<String, Object>) data.get("inputSchema"));
        
        // Parse annotations if present
        if (data.containsKey("annotations")) {
            @SuppressWarnings("unchecked")
            Map<String, Object> annotationsData = (Map<String, Object>) data.get("annotations");
            builder.annotations(ToolAnnotations.fromMap(annotationsData));
        }
        
        return builder.build();
    }

    /**
     * Tool annotations - hints for clients about tool behavior.
     * Per MCP 2025-11-25 spec: MUST be treated as untrusted unless from verified server.
     */
    public static class ToolAnnotations {
        private final String title;
        private final Boolean readOnlyHint;
        private final Boolean destructiveHint;
        private final Boolean idempotentHint;
        private final Boolean openWorldHint;

        private ToolAnnotations(Builder builder) {
            this.title = builder.title;
            this.readOnlyHint = builder.readOnlyHint;
            this.destructiveHint = builder.destructiveHint;
            this.idempotentHint = builder.idempotentHint;
            this.openWorldHint = builder.openWorldHint;
        }

        /** Human-readable title for display */
        public String getTitle() { return title; }
        
        /** Hint that tool does not modify state */
        public Boolean getReadOnlyHint() { return readOnlyHint; }
        
        /** Hint that tool may modify state */
        public Boolean getDestructiveHint() { return destructiveHint; }
        
        /** Hint that multiple identical calls have same effect as single call */
        public Boolean getIdempotentHint() { return idempotentHint; }
        
        /** Hint that tool operates in open world (unknown entities may exist) */
        public Boolean getOpenWorldHint() { return openWorldHint; }

        public static Builder builder() { return new Builder(); }

        @SuppressWarnings("unchecked")
        public static ToolAnnotations fromMap(Map<String, Object> data) {
            if (data == null) return null;
            Builder builder = new Builder();
            if (data.containsKey("title")) builder.title((String) data.get("title"));
            if (data.containsKey("readOnlyHint")) builder.readOnlyHint((Boolean) data.get("readOnlyHint"));
            if (data.containsKey("destructiveHint")) builder.destructiveHint((Boolean) data.get("destructiveHint"));
            if (data.containsKey("idempotentHint")) builder.idempotentHint((Boolean) data.get("idempotentHint"));
            if (data.containsKey("openWorldHint")) builder.openWorldHint((Boolean) data.get("openWorldHint"));
            return builder.build();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new java.util.HashMap<>();
            if (title != null) map.put("title", title);
            if (readOnlyHint != null) map.put("readOnlyHint", readOnlyHint);
            if (destructiveHint != null) map.put("destructiveHint", destructiveHint);
            if (idempotentHint != null) map.put("idempotentHint", idempotentHint);
            if (openWorldHint != null) map.put("openWorldHint", openWorldHint);
            return map;
        }

        public static class Builder {
            private String title;
            private Boolean readOnlyHint;
            private Boolean destructiveHint;
            private Boolean idempotentHint;
            private Boolean openWorldHint;

            public Builder title(String title) { this.title = title; return this; }
            public Builder readOnlyHint(Boolean readOnlyHint) { this.readOnlyHint = readOnlyHint; return this; }
            public Builder destructiveHint(Boolean destructiveHint) { this.destructiveHint = destructiveHint; return this; }
            public Builder idempotentHint(Boolean idempotentHint) { this.idempotentHint = idempotentHint; return this; }
            public Builder openWorldHint(Boolean openWorldHint) { this.openWorldHint = openWorldHint; return this; }

            public ToolAnnotations build() { return new ToolAnnotations(this); }
        }
    }

    public static class Builder {
        private String name;
        private String description;
        private Map<String, Object> inputSchema;
        private ToolAnnotations annotations;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        /**
         * JSON Schema 2020-12 object for tool arguments.
         * Per spec: type MUST be "object".
         */
        public Builder inputSchema(Map<String, Object> inputSchema) {
            this.inputSchema = inputSchema;
            return this;
        }

        /**
         * Optional hints about tool behavior.
         * Per spec: MUST be treated as untrusted unless from verified server.
         */
        public Builder annotations(ToolAnnotations annotations) {
            this.annotations = annotations;
            return this;
        }

        public MCPTool build() {
            return new MCPTool(this);
        }
    }
}