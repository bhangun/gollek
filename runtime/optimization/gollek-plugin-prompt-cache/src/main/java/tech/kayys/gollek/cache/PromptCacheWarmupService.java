package tech.kayys.gollek.promptcache;

import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.cache.CacheEntrySerializer;
import tech.kayys.gollek.cache.PromptCacheConfig;
import tech.kayys.gollek.cache.PromptCacheSnapshotReader;
import tech.kayys.gollek.cache.PromptCacheStore;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Observes the Quarkus {@link StartupEvent} and pre-warms the prompt cache
 * when {@code gollek.cache.prompt.warm-on-startup=true}.
 *
 * <h3>Warm-up sources</h3>
 * <ol>
 *   <li>If the active strategy is {@code disk}, the disk store already rebuilds
 *       its index on {@link PromptCacheStore#initialize()} — this service simply
 *       logs the result.</li>
 *   <li>If the strategy is {@code in-process} or {@code redis}, the service
 *       loads a warm-cache snapshot from the configured disk path (if present)
 *       and re-populates the active store.</li>
 * </ol>
 *
 * <h3>Filtered warm-up</h3>
 * {@code gollek.cache.prompt.warm-model-ids} accepts a comma-separated list of
 * model IDs to warm. If empty, all models found in the snapshot are warmed.
 */
@ApplicationScoped
public class PromptCacheWarmupService {

    private static final Logger LOG = Logger.getLogger(PromptCacheWarmupService.class);

    @Inject PromptCacheConfig config;
    @Inject PromptCacheStore  store;
    @Inject CacheEntrySerializer serializer;

    void onStart(@Observes StartupEvent ev) {
        if (!config.enabled() || !config.warmOnStartup()) {
            LOG.debugf("[PromptCacheWarmup] skipped (enabled=%s warmOnStartup=%s)",
                    config.enabled(), config.warmOnStartup());
            return;
        }

        String strategy = config.strategy();
        LOG.infof("[PromptCacheWarmup] starting warm-up for strategy=%s", strategy);

        if ("disk".equals(strategy)) {
            // Disk store already warmed itself during initialize()
            var stats = store.stats();
            LOG.infof("[PromptCacheWarmup] disk store warm-up complete: %d entries loaded",
                    stats.cachedEntries());
            return;
        }

        // For in-process / redis: load from a companion disk snapshot
        warmFromDiskSnapshot();
    }

    private void warmFromDiskSnapshot() {
        var modelFilter = config.warmModelIds()
                .filter(s -> !s.isBlank())
                .map(s -> Arrays.asList(s.split(",")))
                .map(list -> {
                    Set<String> set = new HashSet<>();
                    for (String item : list) {
                        String trimmed = item.trim();
                        if (!trimmed.isEmpty()) set.add(trimmed);
                    }
                    return set;
                })
                .orElse(null);

        LOG.infof("[PromptCacheWarmup] loading snapshot from disk (models=%s)",
                modelFilter != null ? modelFilter : "all");

        var snapshotFile = java.nio.file.Path.of(config.disk().path(), "prompt-cache.dat");
        final long[] warmed = {0};

        PromptCacheSnapshotReader.read(
                snapshotFile,
                serializer,
                (byte) 0x47,
                4096,
                LOG,
                snapshot -> {
                    var entry = snapshot.entry();
                    if (modelFilter != null && !modelFilter.contains(entry.key().modelId())) {
                        return;
                    }
                    if (!config.ttl().isZero()
                            && entry.createdAt().plus(config.ttl()).isBefore(java.time.Instant.now())) {
                        return;
                    }
                    if (entry.tokenCount() < config.minCacheableTokens()) {
                        return;
                    }
                    store.store(entry);
                    warmed[0]++;
                }
        );

        LOG.infof("[PromptCacheWarmup] snapshot warm-up complete (entries=%d, store=%s)",
                warmed[0], config.strategy());
    }
}
