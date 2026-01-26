package tech.kayys.golek.api.tool;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import org.jetbrains.annotations.Nullable;

import java.util.Map;
import java.util.Objects;

/**
 * Definition of a function that can be called by the model.
 */
public final class Function {

    @NotBlank
    private final String name;

    @Nullable
    private final String description;

    @Nullable
    private final Map<String, Object> parameters;

    @JsonCreator
    public Function(
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("parameters") Map<String, Object> parameters) {
        this.name = Objects.requireNonNull(name, "name");
        this.description = description;
        this.parameters = parameters;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        Function function = (Function) o;
        return Objects.equals(name, function.name) &&
                Objects.equals(description, function.description) &&
                Objects.equals(parameters, function.parameters);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, description, parameters);
    }
}
