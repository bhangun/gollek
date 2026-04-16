package tech.kayys.gollek.kernel.paged;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.jboss.logging.Logger;

import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.kvcache.KVCacheConfig;
import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.spi.plugin.PluginContext;
import tech.kayys.gollek.plugin.optimization.config.OptimizationConfig;
import tech.kayys.gollek.plugin.optimization.config.OptimizationRegistry;

/**
 * Plugin wrapper for PagedAttention optimization.
 * <p>
 * This plugin provides:
 * <ul>
 *   <li>Paged KV-Cache management for efficient memory usage</li>
 *   <li>PagedAttention kernel execution (CPU fallback available)</li>
 *   <li>Block table generation for continuous batching</li>
 *   <li>Memory defragmentation support</li>
 * </ul>
 * 
 * <h2>Configuration</h2>
 * <pre>
 * # Enable/disable
 * gollek.optimizations.paged-attention.enabled=true
 * 
 * # Block size (tokens per block)
 * gollek.optimizations.paged-attention.blockSize=16
 * 
 * # Total blocks in pool
 * gollek.optimizations.paged-attention.totalBlocks=2048
 * </pre>
 * 
 * @see PagedKVCacheManager
 * @see PagedAttentionCpuFallback
 */
@ApplicationScoped
public class PagedAttentionPlugin implements GollekPlugin {

    private static final Logger LOG = Logger.getLogger(PagedAttentionPlugin.class);
    private static final String PLUGIN_ID = "paged-attention";

    @Inject
    PagedKVCacheManager kvCacheManager;

    private OptimizationConfig config;

    @Override
    public String id() {
        return "tech.kayys/" + PLUGIN_ID;
    }

    @Override
    public int order() {
        return config != null ? config.priority() : 10;
    }

    @Override
    public void initialize(PluginContext context) {
        // Load configuration
        this.config = OptimizationRegistry.getInstance().getConfig(PLUGIN_ID);
        
        if (config == null) {
            // Register with defaults if not already registered
            OptimizationRegistry.getInstance().register(PLUGIN_ID);
            this.config = OptimizationRegistry.getInstance().getConfig(PLUGIN_ID);
        }

        if (!config.enabled()) {
            LOG.infof("PagedAttentionPlugin is DISABLED by configuration");
            return;
        }

        // Get custom properties
        int blockSize = config.getCustomPropertyAsInt("blockSize", 16);
        int totalBlocks = config.getCustomPropertyAsInt("totalBlocks", 2048);
        
        LOG.infof("PagedAttentionPlugin initialized: blockSize=%d, totalBlocks=%d",
            blockSize, totalBlocks);
    }

    @Override
    public void shutdown() {
        if (kvCacheManager != null) {
            kvCacheManager.close();
        }
        LOG.info("PagedAttentionPlugin destroyed");
    }

    @Override
    public boolean isHealthy() {
        // If disabled, still report healthy
        if (config != null && !config.enabled()) {
            return true;
        }
        return kvCacheManager != null;
    }

    /**
     * Checks if this plugin is currently enabled.
     */
    public boolean isEnabled() {
        return config != null && config.enabled();
    }

    /**
     * Gets the current configuration.
     */
    public OptimizationConfig getConfig() {
        return config;
    }

    /**
     * Gets the KV cache manager instance (may be null if disabled).
     */
    public PagedKVCacheManager getKVCacheManager() {
        return kvCacheManager;
    }

    /**
     * Creates a new KV cache configuration with recommended defaults
     * or custom values from plugin config.
     */
    public KVCacheConfig createDefaultConfig(int numHeads, int headDim) {
        if (config == null) {
            // Fallback defaults
            return KVCacheConfig.builder()
                .totalBlocks(2048)
                .blockSize(16)
                .numHeads(numHeads)
                .headDim(headDim)
                .build();
        }

        int totalBlocks = config.getCustomPropertyAsInt("totalBlocks", 2048);
        int blockSize = config.getCustomPropertyAsInt("blockSize", 16);

        return KVCacheConfig.builder()
            .totalBlocks(totalBlocks)
            .blockSize(blockSize)
            .numHeads(numHeads)
            .headDim(headDim)
            .build();
    }
}
