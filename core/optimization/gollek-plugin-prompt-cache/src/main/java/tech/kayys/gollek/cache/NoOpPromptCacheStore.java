package tech.kayys.gollek.cache;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.cache.CacheStrategy;
import tech.kayys.gollek.cache.CachedKVEntry;
import tech.kayys.gollek.cache.PrefixHash;
import tech.kayys.gollek.cache.PromptCacheStats;
import tech.kayys.gollek.cache.PromptCacheStore;

import java.util.Optional;

/**
 * No-operation store — always returns empty on lookup, discards stores.
 * Used when {@code gollek.cache.prompt.enabled=false} or strategy is {@code noop}.
 * Zero allocation, zero overhead — safe to inject unconditionally.
 */
@ApplicationScoped
@CacheStrategy("noop")
public class NoOpPromptCacheStore implements PromptCacheStore {

    @Override
    public Optional<CachedKVEntry> lookup(PrefixHash hash) {
        return Optional.empty();
    }

    @Override
    public void store(CachedKVEntry entry) {
        // intentional no-op
    }

    @Override
    public void invalidateByModel(String modelId)   {}
    @Override
    public void invalidateBySession(String sessionId) {}
    @Override
    public void invalidateAll()                       {}

    @Override
    public PromptCacheStats stats() {
        return PromptCacheStats.empty("noop");
    }

    @Override
    public String strategyName() { return "noop"; }
}
