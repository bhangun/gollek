package tech.kayys.golek.engine.model;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Configuration for model runner initialization.
 *
 * @author Bhangun
 * @since 1.0.0
 */
public class RunnerConfiguration {

    /**
     * Configuration parameters.
     */
    private Map<String, Object> parameters = new HashMap<>();

    public RunnerConfiguration() {}

    public RunnerConfiguration(Map<String, Object> parameters) {
        this.parameters = parameters != null ? parameters : new HashMap<>();
    }

    /**
     * Get configuration parameter with type casting.
     */
    @SuppressWarnings("unchecked")
    public <T> T getParameter(String key, Class<T> type, T defaultValue) {
        Object value = parameters.get(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return (T) value;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Get integer parameter.
     */
    public Integer getIntParameter(String key, Integer defaultValue) {
        return getParameter(key, Integer.class, defaultValue);
    }

    /**
     * Get string parameter.
     */
    public String getStringParameter(String key, String defaultValue) {
        return getParameter(key, String.class, defaultValue);
    }

    /**
     * Get boolean parameter.
     */
    public Boolean getBooleanParameter(String key, Boolean defaultValue) {
        return getParameter(key, Boolean.class, defaultValue);
    }

    // Getters
    public Map<String, Object> getParameters() {
        return parameters;
    }

    // Setters
    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters != null ? parameters : new HashMap<>();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RunnerConfiguration that = (RunnerConfiguration) o;
        return Objects.equals(parameters, that.parameters);
    }

    @Override
    public int hashCode() {
        return Objects.hash(parameters);
    }
    
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Map<String, Object> parameters = new HashMap<>();

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters = parameters != null ? parameters : new HashMap<>();
            return this;
        }

        public Builder putParameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        public RunnerConfiguration build() {
            return new RunnerConfiguration(parameters);
        }
    }
}
