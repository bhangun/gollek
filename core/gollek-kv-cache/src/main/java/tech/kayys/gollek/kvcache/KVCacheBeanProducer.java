package tech.kayys.gollek.kvcache;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Singleton;
import org.jboss.logging.Logger;

/**
 * CDI producer for the Paged KV-Cache Manager.
 * <p>
 * Reads configuration from the Quarkus config system and creates
 * the singleton {@link PagedKVCacheManager} instance.
 * <p>
 * The manager is created lazily on first injection and shared
 * across the application.
 */
@ApplicationScoped
public class KVCacheBeanProducer {

    private static final Logger LOG = Logger.getLogger(KVCacheBeanProducer.class);

    @Produces
    @Singleton
    public KVCacheConfig kvCacheConfig() {
        // Default configuration — will be overridden by Quarkus config in production
        KVCacheConfig config = KVCacheConfig.builder()
                .blockSize(16)
                .totalBlocks(1024)
                .numLayers(32)
                .numHeads(32)
                .headDim(128)
                .useGpu(false)
                .maxBlocksPerRequest(0)
                .build();

        LOG.infof("KV-Cache config: %s", config);
        return config;
    }

    @Produces
    @Singleton
    public PagedKVCacheManager pagedKVCacheManager(KVCacheConfig config) {
        LOG.infof("Initializing Paged KV-Cache Manager with %d blocks of %d tokens each " +
                  "(total pool: %d MB)",
                config.getTotalBlocks(),
                config.getBlockSize(),
                config.totalPoolBytes() / (1024 * 1024));

        PagedKVCacheManager manager = new PagedKVCacheManager(config);

        LOG.infof("KV-Cache Manager ready: %s", manager);
        return manager;
    }
}
