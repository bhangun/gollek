package tech.kayys.golek.api.tool;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.util.*;

/**
 * MCP-compliant tool/function definition.
 * Supports both function calling and MCP tool protocols.
 */
public final class ToolDefinition {

    public enum Type {
        FUNCTION, // OpenAI-style function calling
        MCP_TOOL, // MCP tool protocol
        CODE_INTERPRETER,
        FILE_SEARCH
    }

    @NotBlank
    private final String name;

    private final Type type;
    private final String description;
    private final Map<String, Object> parameters;
    private final boolean strict;
    private final Map<String, Object> metadata;

    @JsonCreator
    public ToolDefinition(
            @JsonProperty("name") String name,
            @JsonProperty("type") Type type,
            @JsonProperty("description") String description,
            @JsonProperty("parameters") Map<String, Object> parameters,
            @JsonProperty("strict") boolean strict,
            @JsonProperty("metadata") Map<String, Object> metadata) {
        this.name = Objects.requireNonNull(name, "name");
        this.type = type != null ? type : Type.FUNCTION;
        this.description = description;
        this.parameters = parameters != null
                ? Collections.unmodifiableMap(new HashMap<>(parameters))
                : Collections.emptyMap();
        this.strict = strict;
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
    }

    // Getters
    public String getName() {
        return name;
    }

    public Type getType() {
        return type;
    }

    public Optional<String> getDescription() {
        return Optional.ofNullable(description);
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public boolean isStrict() {
        return strict;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String name;
        private Type type = Type.FUNCTION;
        private String description;
        private final Map<String, Object> parameters = new HashMap<>();
        private boolean strict = false;
        private final Map<String, Object> metadata = new HashMap<>();

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder type(Type type) {
            this.type = type;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters.putAll(parameters);
            return this;
        }

        public Builder parameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        public Builder strict(boolean strict) {
            this.strict = strict;
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public ToolDefinition build() {
            Objects.requireNonNull(name, "name is required");
            return new ToolDefinition(
                    name, type, description, parameters, strict, metadata);
        }
    }

    /**
     * Create MCP-compatible tool definition from JSON Schema
     */
    public static ToolDefinition fromMCPSchema(
            String name,
            String description,
            Map<String, Object> inputSchema) {
        return builder()
                .name(name)
                .type(Type.MCP_TOOL)
                .description(description)
                .parameters(inputSchema)
                .build();
    }

    @Override
    public String toString() {
        return "ToolDefinition{" +
                "name='" + name + '\'' +
                ", type=" + type +
                ", description='" + description + '\'' +
                '}';
    }
}