package tech.kayys.gollek.promptcache;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.jboss.logging.Logger;
import tech.kayys.gollek.cache.PromptCacheConfig;
import tech.kayys.gollek.cache.CacheStrategy;
import tech.kayys.gollek.cache.PromptCacheStore;

/**
 * CDI producer that selects the active {@link PromptCacheStore} implementation
 * based on {@code gollek.cache.prompt.strategy}.
 *
 * <p>Mirrors the pattern of {@code KVCacheBeanProducer} in the kv-cache module.
 * Only the produced {@code @Singleton} bean is directly injected by plugins and
 * services — implementation classes are never injected directly.
 *
 * <p>Strategy resolution order:
 * <ol>
 *   <li>If {@code enabled=false} → noop (zero overhead)</li>
 *   <li>{@code "redis"}       → {@link tech.kayys.gollek.promptcache.store.RedisPromptCacheStore}</li>
 *   <li>{@code "disk"}        → {@link tech.kayys.gollek.promptcache.store.DiskPromptCacheStore}</li>
 *   <li>{@code "in-process"}  → {@link tech.kayys.gollek.promptcache.store.InProcessPromptCacheStore} (default)</li>
 * </ol>
 */
@ApplicationScoped
public class PromptCacheBeanProducer {

    private static final Logger LOG = Logger.getLogger(PromptCacheBeanProducer.class);

    @Inject
    PromptCacheConfig config;

    @Inject
    @CacheStrategy("in-process")
    Instance<PromptCacheStore> inProcess;

    @Inject
    @CacheStrategy("redis")
    Instance<PromptCacheStore> redis;

    @Inject
    @CacheStrategy("disk")
    Instance<PromptCacheStore> disk;

    @Inject
    @CacheStrategy("noop")
    Instance<PromptCacheStore> noop;

    @Produces
    @Singleton
    public PromptCacheStore promptCacheStore() {
        if (!config.enabled()) {
            LOG.info("[PromptCache] disabled — producing noop store");
            return noop.get();
        }

        PromptCacheStore store = switch (config.strategy()) {
            case "redis"      -> redis.get();
            case "disk"       -> disk.get();
            case "noop"       -> noop.get();
            default           -> inProcess.get();
        };

        store.initialize();

        LOG.infof("[PromptCache] active strategy=%s scope=%s ttl=%s eviction=%s",
                store.strategyName(),
                config.scope(),
                config.ttl(),
                config.evictionPolicy());

        return store;
    }
}
