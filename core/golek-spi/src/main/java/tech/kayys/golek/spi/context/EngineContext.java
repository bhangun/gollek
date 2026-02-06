package tech.kayys.golek.spi.context;

import tech.kayys.golek.spi.plugin.PluginRegistry;
import tech.kayys.golek.spi.provider.ProviderRegistry;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.time.Instant;

/**
 * Global engine context providing access to kernel services.
 * Immutable and thread-safe.
 * 
 * This is the read-only view of engine state available to plugins.
 */
public interface EngineContext {

    /**
     * Get plugin registry
     */
    PluginRegistry pluginRegistry();

    /**
     * Get provider registry
     */
    ProviderRegistry providerRegistry();

    /**
     * Get shared executor service for async operations
     */
    ExecutorService executorService();

    /**
     * Get engine configuration (read-only)
     */
    Map<String, Object> config();

    /**
     * Get config value by key
     */
    <T> T getConfig(String key, Class<T> type);

    /**
     * Get config value with default
     */
    <T> T getConfig(String key, T defaultValue);

    /**
     * Check if engine is running
     */
    boolean isRunning();

    /**
     * Get engine start time
     */
    Instant startTime();

    /**
     * Get engine version
     */
    String version();
}
