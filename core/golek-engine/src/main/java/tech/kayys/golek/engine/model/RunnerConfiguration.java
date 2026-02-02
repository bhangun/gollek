package tech.kayys.golek.engine.model;

import java.util.HashMap;
import java.util.Map;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Configuration for model runner initialization.
 * 
 * @author bhangun
 * @since 1.0.0
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RunnerConfiguration {

    /**
     * Configuration parameters.
     */
    @Builder.Default
    private Map<String, Object> parameters = new HashMap<>();

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
}
