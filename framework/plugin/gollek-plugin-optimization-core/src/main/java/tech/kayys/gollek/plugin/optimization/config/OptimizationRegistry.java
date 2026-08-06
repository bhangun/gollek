package tech.kayys.gollek.plugin.optimization.config;

import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Registry for optimization plugin configurations.
 */
public class OptimizationRegistry {

    private static final Logger LOG = Logger.getLogger(OptimizationRegistry.class);
    private static volatile OptimizationRegistry instance;

    private final Map<String, OptimizationConfig> configurations = new ConcurrentHashMap<>();
    private final Map<String, Boolean> enabledState = new ConcurrentHashMap<>();
    private final List<OptimizationStateListener> listeners = new CopyOnWriteArrayList<>();

    private OptimizationRegistry() {
        registerDefaults();
    }

    public static OptimizationRegistry getInstance() {
        if (instance == null) {
            synchronized (OptimizationRegistry.class) {
                if (instance == null) {
                    instance = new OptimizationRegistry();
                }
            }
        }
        return instance;
    }

    public static void reset() { instance = null; }

    public void register(String pluginId) {
        register(OptimizationConfig.enabled(pluginId));
    }

    public void register(OptimizationConfig config) {
        configurations.put(config.pluginId(), config);
        enabledState.put(config.pluginId(), config.enabled());
        LOG.infof("Registered optimization: %s (enabled=%s, priority=%d)",
            config.pluginId(), config.enabled(), config.priority());
    }

    public void unregister(String pluginId) {
        configurations.remove(pluginId);
        enabledState.remove(pluginId);
    }

    public void setEnabled(String pluginId, boolean enabled) {
        Boolean previous = enabledState.put(pluginId, enabled);
        if (previous == null || previous != enabled) {
            LOG.infof("Optimization %s %s", pluginId, enabled ? "ENABLED" : "DISABLED");
            notifyListeners(pluginId, enabled);
        }
    }

    public boolean isEnabled(String pluginId) {
        return enabledState.getOrDefault(pluginId, false);
    }

    public boolean isActive(String pluginId, String deviceType) {
        OptimizationConfig config = configurations.get(pluginId);
        if (config == null) return false;
        if (!isEnabled(pluginId)) return false;
        return config.matchesDevice(deviceType);
    }

    public OptimizationConfig getConfig(String pluginId) {
        return configurations.get(pluginId);
    }

    public Map<String, OptimizationConfig> getAllConfigs() {
        return Collections.unmodifiableMap(configurations);
    }

    public List<OptimizationConfig> getActiveOptimizations(String deviceType) {
        return configurations.values().stream()
            .filter(config -> isEnabled(config.pluginId()))
            .filter(config -> config.matchesDevice(deviceType))
            .sorted(Comparator.comparingInt(OptimizationConfig::priority))
            .toList();
    }

    public void addListener(OptimizationStateListener listener) { listeners.add(listener); }
    public void removeListener(OptimizationStateListener listener) { listeners.remove(listener); }

    private void notifyListeners(String pluginId, boolean enabled) {
        for (OptimizationStateListener listener : listeners) {
            try { listener.onStateChanged(pluginId, enabled); }
            catch (Exception e) { LOG.warnf(e, "Error notifying listener: %s", pluginId); }
        }
    }

    private void registerDefaults() {
        register(OptimizationConfig.builder("paged-attention").priority(10)
            .customProperty("blockSize", "16").customProperty("totalBlocks", "2048").build());
        register(OptimizationConfig.builder("kv-cache").priority(20)
            .customProperty("evictionPolicy", "LRU").customProperty("maxSequenceLength", "32768").build());
        register(OptimizationConfig.builder("continuous-batching").priority(15)
            .customProperty("maxBatchSize", "256").customProperty("scheduleIntervalMs", "10").build());
        register(OptimizationConfig.builder("flash-attention").priority(5)
            .addDeviceFilter("cuda").addDeviceFilter("metal").customProperty("minSeqLen", "512").build());
        register(OptimizationConfig.builder("prompt-cache").priority(25)
            .customProperty("maxPrompts", "1000").customProperty("ttlMinutes", "60").build());
        register(OptimizationConfig.builder("fa3").priority(3)
            .addDeviceFilter("cuda").customProperty("minComputeCapability", "90").build());
        register(OptimizationConfig.builder("fa4").priority(2)
            .addDeviceFilter("cuda").addDeviceFilter("metal").customProperty("minComputeCapability", "100").build());
    }

    public interface OptimizationStateListener {
        void onStateChanged(String pluginId, boolean enabled);
    }
}
