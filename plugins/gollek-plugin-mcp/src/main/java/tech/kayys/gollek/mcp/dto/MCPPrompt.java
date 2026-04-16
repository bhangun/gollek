package tech.kayys.gollek.mcp.dto;

import java.util.List;
import java.util.Map;

/**
 * Represents an MCP prompt.
 * Compliant with MCP specification 2025-11-25.
 * 
 * <p>Prompts provide templated messages and workflows for users.
 * Prompts can accept arguments to customize the prompt behavior.
 */
public class MCPPrompt {
    private final String name;
    private final String description;
    private final List<PromptArgument> arguments;

    private MCPPrompt(Builder builder) {
        this.name = builder.name;
        this.description = builder.description;
        this.arguments = builder.arguments;
    }

    /** Unique prompt name */
    public String getName() { return name; }
    
    /** Optional description */
    public String getDescription() { return description; }
    
    /** List of arguments the prompt accepts */
    public List<PromptArgument> getArguments() { return arguments; }

    public static Builder builder() {
        return new Builder();
    }

    @SuppressWarnings("unchecked")
    public static MCPPrompt fromMap(Map<String, Object> data) {
        Builder builder = new Builder()
                .name((String) data.get("name"))
                .description((String) data.get("description"));
        
        // Parse arguments - can be List<String> or List<Map<String, Object>>
        if (data.containsKey("arguments")) {
            Object argsObj = data.get("arguments");
            if (argsObj instanceof List<?> argsList) {
                if (!argsList.isEmpty()) {
                    Object first = argsList.get(0);
                    if (first instanceof String) {
                        // Simple string arguments
                        builder.arguments(argsList.stream()
                                .map(arg -> PromptArgument.builder()
                                        .name((String) arg)
                                        .build())
                                .toList());
                    } else if (first instanceof Map) {
                        // Structured arguments
                        builder.arguments(((List<Map<String, Object>>) argsList).stream()
                                .map(PromptArgument::fromMap)
                                .toList());
                    }
                }
            }
        }
        
        return builder.build();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new java.util.HashMap<>();
        map.put("name", name);
        if (description != null) map.put("description", description);
        if (arguments != null && !arguments.isEmpty()) {
            map.put("arguments", arguments.stream().map(PromptArgument::toMap).toList());
        }
        return map;
    }

    /**
     * Prompt argument definition.
     * Per MCP 2025-11-25 spec.
     */
    public static class PromptArgument {
        private final String name;
        private final String description;
        private final boolean required;

        private PromptArgument(Builder builder) {
            this.name = builder.name;
            this.description = builder.description;
            this.required = builder.required;
        }

        /** Argument name */
        public String getName() { return name; }
        
        /** Optional description */
        public String getDescription() { return description; }
        
        /** Whether this argument must be provided */
        public boolean isRequired() { return required; }

        public static Builder builder() { return new Builder(); }

        public static PromptArgument fromMap(Map<String, Object> data) {
            if (data == null) return null;
            return new Builder()
                    .name((String) data.get("name"))
                    .description((String) data.get("description"))
                    .required(Boolean.TRUE.equals(data.get("required")))
                    .build();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new java.util.HashMap<>();
            map.put("name", name);
            if (description != null) map.put("description", description);
            if (required) map.put("required", true);
            return map;
        }

        public static class Builder {
            private String name;
            private String description;
            private boolean required;

            public Builder name(String name) { this.name = name; return this; }
            public Builder description(String description) { this.description = description; return this; }
            public Builder required(boolean required) { this.required = required; return this; }

            public PromptArgument build() { return new PromptArgument(this); }
        }
    }

    public static class Builder {
        private String name;
        private String description;
        private List<PromptArgument> arguments;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder arguments(List<PromptArgument> arguments) {
            this.arguments = arguments;
            return this;
        }

        /** Support legacy string argument lists */
        public Builder argumentsFromStrings(List<String> argumentNames) {
            if (argumentNames == null) {
                this.arguments = null;
            } else {
                this.arguments = argumentNames.stream()
                        .map(name -> PromptArgument.builder().name(name).build())
                        .toList();
            }
            return this;
        }

        public MCPPrompt build() {
            return new MCPPrompt(this);
        }
    }
}