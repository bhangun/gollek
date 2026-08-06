package tech.kayys.gollek.promptcache;

import io.quarkus.runtime.annotations.RegisterForReflection;
import tech.kayys.gollek.cache.CachedKVEntry;
import tech.kayys.gollek.cache.PrefixHash;
import tech.kayys.gollek.cache.PromptCacheStats;

/**
 * GraalVM native-image reflection registration for prompt-cache types.
 *
 * <p>Mirrors the pattern of {@code NativeImageFeature} in {@code gollek-engine-core}.
 * Required for serialization of records (Jackson, JSON-B) in native mode.
 */
@RegisterForReflection(targets = {
        CachedKVEntry.class,
        PrefixHash.class,
        PromptCacheStats.class
})
public class NativeImagePromptCacheFeature {
    // No body needed — annotation drives native-image registration
}
