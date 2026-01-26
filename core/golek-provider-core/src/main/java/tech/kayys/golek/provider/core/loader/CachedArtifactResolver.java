package tech.kayys.golek.provider.core.loader;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Objects;

/**
 * Caching decorator for artifact resolvers
 */
public class CachedArtifactResolver implements ArtifactResolver {

    private static final Logger LOG = Logger.getLogger(CachedArtifactResolver.class);

    private final ArtifactResolver delegate;
    private final Cache<String, Path> cache;

    public CachedArtifactResolver(ArtifactResolver delegate, int maxSize, Duration ttl) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.cache = Caffeine.newBuilder()
                .maximumSize(maxSize)
                .expireAfterWrite(ttl)
                .recordStats()
                .build();
    }

    @Override
    public Uni<Path> resolve(String artifactId) {
        Path cached = cache.getIfPresent(artifactId);
        if (cached != null) {
            LOG.debugf("Cache hit for artifact: %s", artifactId);
            return Uni.createFrom().item(cached);
        }

        LOG.debugf("Cache miss for artifact: %s", artifactId);
        return delegate.resolve(artifactId)
                .invoke(path -> cache.put(artifactId, path));
    }

    @Override
    public boolean isAvailableLocally(String artifactId) {
        return cache.getIfPresent(artifactId) != null ||
                delegate.isAvailableLocally(artifactId);
    }

    @Override
    public Path getLocalPath(String artifactId) {
        Path cached = cache.getIfPresent(artifactId);
        return cached != null ? cached : delegate.getLocalPath(artifactId);
    }

    @Override
    public Uni<Void> clearCache(String artifactId) {
        return Uni.createFrom().item(() -> {
            cache.invalidate(artifactId);
            return null;
        }).chain(() -> delegate.clearCache(artifactId));
    }

    public void clearAll() {
        cache.invalidateAll();
        LOG.info("Cleared all cached artifacts");
    }

    public long cacheSize() {
        return cache.estimatedSize();
    }

    public double hitRate() {
        return cache.stats().hitRate();
    }
}