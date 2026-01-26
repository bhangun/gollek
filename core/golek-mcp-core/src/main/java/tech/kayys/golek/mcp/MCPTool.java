package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.util.*;

/**
 * Represents an MCP tool definition.
 * Immutable and serializable.
 */
public final class MCPTool {

    @NotBlank
    private final String name;

    private final String description;
    private final Map<String, Object> inputSchema;
    private final Map<String, Object> metadata;

    @JsonCreator
    public MCPTool(
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("inputSchema") Map<String, Object> inputSchema,
            @JsonProperty("metadata") Map<String, Object> metadata) {
        this.name = Objects.requireNonNull(name, "name");
        this.description = description;
        this.inputSchema = inputSchema != null
                ? Collections.unmodifiableMap(new HashMap<>(inputSchema))
                : Collections.emptyMap();
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
    }

    // Getters
    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public Map<String, Object> getInputSchema() {
        return inputSchema;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Create from MCP protocol map
     */
    public static MCPTool fromMap(Map<String, Object> data) {
        return new MCPTool(
                (String) data.get("name"),
                (String) data.get("description"),
                (Map<String, Object>) data.get("inputSchema"),
                (Map<String, Object>) data.get("metadata"));
    }

    /**
     * Validate arguments against schema
     */
    public boolean validateArguments(Map<String, Object> arguments) {
        if (inputSchema.isEmpty()) {
            return true; // No schema means no validation
        }

        // JSON Schema validation logic
        @SuppressWarnings("unchecked")
        Map<String, Object> properties = (Map<String, Object>) inputSchema.get("properties");
        if (properties == null) {
            return true;
        }

        @SuppressWarnings("unchecked")
        List<String> required = (List<String>) inputSchema.get("required");
        if (required != null) {
            for (String field : required) {
                if (!arguments.containsKey(field)) {
                    return false;
                }
            }
        }

        return true;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof MCPTool mcpTool))
            return false;
        return name.equals(mcpTool.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }

    @Override
    public String toString() {
        return "MCPTool{name='" + name + "', description='" + description + "'}";
    }
}