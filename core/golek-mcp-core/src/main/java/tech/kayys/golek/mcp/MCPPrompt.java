package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.util.*;

/**
 * Represents an MCP prompt template.
 * Immutable and serializable.
 */
public final class MCPPrompt {

    @NotBlank
    private final String name;

    private final String description;
    private final List<PromptArgument> arguments;
    private final Map<String, Object> metadata;

    @JsonCreator
    public MCPPrompt(
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("arguments") List<PromptArgument> arguments,
            @JsonProperty("metadata") Map<String, Object> metadata) {
        this.name = Objects.requireNonNull(name, "name");
        this.description = description;
        this.arguments = arguments != null
                ? Collections.unmodifiableList(new ArrayList<>(arguments))
                : Collections.emptyList();
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
    }

    // Prompt argument definition
    public record PromptArgument(
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("required") boolean required) {
        public PromptArgument {
            Objects.requireNonNull(name, "name");
        }
    }

    // Getters
    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public List<PromptArgument> getArguments() {
        return arguments;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Create from MCP protocol map
     */
    public static MCPPrompt fromMap(Map<String, Object> data) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> argList = (List<Map<String, Object>>) data.get("arguments");
        List<PromptArgument> arguments = null;

        if (argList != null) {
            arguments = argList.stream()
                    .map(argData -> new PromptArgument(
                            (String) argData.get("name"),
                            (String) argData.get("description"),
                            Boolean.TRUE.equals(argData.get("required"))))
                    .toList();
        }

        return new MCPPrompt(
                (String) data.get("name"),
                (String) data.get("description"),
                arguments,
                (Map<String, Object>) data.get("metadata"));
    }

    /**
     * Validate arguments
     */
    public boolean validateArguments(Map<String, String> providedArgs) {
        for (PromptArgument arg : arguments) {
            if (arg.required() && !providedArgs.containsKey(arg.name())) {
                return false;
            }
        }
        return true;
    }

    /**
     * Get required arguments
     */
    public List<String> getRequiredArguments() {
        return arguments.stream()
                .filter(PromptArgument::required)
                .map(PromptArgument::name)
                .toList();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof MCPPrompt mcpPrompt))
            return false;
        return name.equals(mcpPrompt.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }

    @Override
    public String toString() {
        return "MCPPrompt{name='" + name + "', description='" + description + "'}";
    }
}