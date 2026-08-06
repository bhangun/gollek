package tech.kayys.gollek.plugin.optimization.config;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Unified configuration for Gollek optimization plugins.
 * <p>
 * All optimization plugins support these standard configuration options:
 * <ul>
 *   <li><b>enabled</b> - Whether the optimization is active (default: true)</li>
 *   <li><b>priority</b> - Execution order, lower = earlier (default: 100)</li>
 *   <li><b>deviceFilter</b> - GPU architectures to target (default: all)</li>
 *   <li><b>threshold</b> - Performance threshold for activation (default: 0)</li>
 * </ul>
 * 
 * @since 0.1.0
 */
public final class OptimizationConfig {

    /** Plugin identifier */
    private final String pluginId;

    /** Whether the optimization is enabled (default: true) */
    private final boolean enabled;

    /** Execution priority - lower numbers execute first (default: 100) */
    private final int priority;

    /** GPU device types to target (empty = all devices) */
    private final java.util.List<String> deviceFilter;

    /** Minimum sequence length to activate optimization (default: 0) */
    private final int minSequenceLength;

    /** Maximum memory overhead in MB (0 = unlimited) */
    private final long maxMemoryOverheadMB;

    /** Custom plugin-specific properties */
    private final Map<String, String> customProperties;

    private OptimizationConfig(Builder builder) {
        this.pluginId = Objects.requireNonNull(builder.pluginId, "pluginId is required");
        this.enabled = builder.enabled;
        this.priority = builder.priority;
        this.deviceFilter = java.util.List.copyOf(builder.deviceFilter);
        this.minSequenceLength = builder.minSequenceLength;
        this.maxMemoryOverheadMB = builder.maxMemoryOverheadMB;
        this.customProperties = Map.copyOf(builder.customProperties);
    }

    public static Builder builder(String pluginId) {
        return new Builder(pluginId);
    }

    public static OptimizationConfig enabled(String pluginId) {
        return builder(pluginId).build();
    }

    public static OptimizationConfig disabled(String pluginId) {
        return builder(pluginId).enabled(false).build();
    }

    public String pluginId() { return pluginId; }
    public boolean enabled() { return enabled; }
    public int priority() { return priority; }
    public java.util.List<String> deviceFilter() { return deviceFilter; }
    public int minSequenceLength() { return minSequenceLength; }
    public long maxMemoryOverheadMB() { return maxMemoryOverheadMB; }
    public Map<String, String> customProperties() { return customProperties; }

    public String getCustomProperty(String key, String defaultValue) {
        return customProperties.getOrDefault(key, defaultValue);
    }

    public int getCustomPropertyAsInt(String key, int defaultValue) {
        String value = customProperties.get(key);
        if (value == null) return defaultValue;
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public boolean getCustomPropertyAsBoolean(String key, boolean defaultValue) {
        String value = customProperties.get(key);
        if (value == null) return defaultValue;
        return Boolean.parseBoolean(value);
    }

    public boolean matchesDevice(String deviceType) {
        if (deviceFilter.isEmpty()) return true;
        return deviceFilter.contains(deviceType.toLowerCase());
    }

    @Override
    public String toString() {
        return String.format("OptimizationConfig{plugin='%s', enabled=%s, priority=%d, devices=%s}",
            pluginId, enabled, priority, deviceFilter.isEmpty() ? "all" : deviceFilter);
    }

    public static final class Builder {
        private final String pluginId;
        private boolean enabled = true;
        private int priority = 100;
        private final java.util.List<String> deviceFilter = new java.util.ArrayList<>();
        private int minSequenceLength = 0;
        private long maxMemoryOverheadMB = 0;
        private final Map<String, String> customProperties = new HashMap<>();

        private Builder(String pluginId) {
            this.pluginId = pluginId;
        }

        public Builder enabled(boolean enabled) { this.enabled = enabled; return this; }
        public Builder priority(int priority) { this.priority = priority; return this; }
        public Builder deviceFilter(java.util.List<String> deviceFilter) {
            this.deviceFilter.clear();
            this.deviceFilter.addAll(deviceFilter);
            return this;
        }
        public Builder addDeviceFilter(String deviceType) {
            this.deviceFilter.add(deviceType.toLowerCase());
            return this;
        }
        public Builder minSequenceLength(int minSequenceLength) { this.minSequenceLength = minSequenceLength; return this; }
        public Builder maxMemoryOverheadMB(long maxMemoryOverheadMB) { this.maxMemoryOverheadMB = maxMemoryOverheadMB; return this; }
        public Builder customProperty(String key, String value) {
            this.customProperties.put(key, value);
            return this;
        }
        public Builder fromMap(Map<String, String> properties) {
            this.customProperties.putAll(properties);
            return this;
        }
        public OptimizationConfig build() {
            return new OptimizationConfig(this);
        }
    }
}
