package tech.kayys.gollek.cache;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tag;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.cache.PromptCacheConfig;

import java.time.Duration;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Micrometer-based metrics for the prompt-cache subsystem.
 *
 * <h3>Exposed meters</h3>
 * <pre>
 * gollek_prompt_cache_lookups_total{strategy, result}   Counter  — total lookups (result=hit|miss)
 * gollek_prompt_cache_stores_total{strategy}             Counter  — entries stored
 * gollek_prompt_cache_evictions_total{strategy}          Counter  — entries evicted
 * gollek_prompt_cache_saved_tokens_total{strategy}       Counter  — tokens saved by cache hits
 * gollek_prompt_cache_saved_tokens{strategy}             Summary  — token saving distribution per request
 * gollek_prompt_cache_entries{strategy}                  Gauge    — current entry count (supplier)
 * gollek_prompt_cache_hit_rate{strategy}                 Gauge    — rolling 5-min window hit rate
 * gollek_prompt_cache_lookup_duration{strategy, result}  Timer    — time spent in lookup (ns precision)
 * </pre>
 *
 * <p>All meters are tagged with {@code strategy} (e.g. {@code in-process}, {@code redis}, {@code disk})
 * for easy dashboarding across deployment modes.
 */
@ApplicationScoped
public class PromptCacheMetrics {

    private static final Logger LOG = Logger.getLogger(PromptCacheMetrics.class);

    private static final String PREFIX = "gollek.prompt_cache";

    private final Counter hitsCounter;
    private final Counter missesCounter;
    private final Counter storesCounter;
    private final Counter evictionsCounter;
    private final Counter savedTokensCounter;

    private final DistributionSummary savedTokensSummary;

    private final Timer hitLookupTimer;
    private final Timer missLookupTimer;

    // Rolling 5-minute hit-rate window (ring buffer of per-second counts)
    private static final int WINDOW_SECONDS = 300;
    private final long[] hitWindow  = new long[WINDOW_SECONDS];
    private final long[] totalWindow = new long[WINDOW_SECONDS];
    private final AtomicLong windowPtr = new AtomicLong(0);
    private final AtomicLong lastSecond = new AtomicLong(System.currentTimeMillis() / 1000);

    @Inject
    public PromptCacheMetrics(MeterRegistry registry, PromptCacheConfig config) {
        String strategy = config.strategy();
        Tags tags = Tags.of(Tag.of("strategy", strategy));

        this.hitsCounter = registry.counter(PREFIX + ".lookups.total",
                tags.and("result", "hit"));
        this.missesCounter = registry.counter(PREFIX + ".lookups.total",
                tags.and("result", "miss"));
        this.storesCounter = registry.counter(PREFIX + ".stores.total", tags);
        this.evictionsCounter = registry.counter(PREFIX + ".evictions.total", tags);
        this.savedTokensCounter = registry.counter(PREFIX + ".saved_tokens.total", tags);

        this.savedTokensSummary = DistributionSummary.builder(PREFIX + ".saved_tokens")
                .tags(tags)
                .description("Distribution of tokens saved per cache hit")
                .publishPercentileHistogram()
                .register(registry);

        this.hitLookupTimer = Timer.builder(PREFIX + ".lookup.duration")
                .tags(tags.and("result", "hit"))
                .description("Time spent in cache lookup resulting in a hit")
                .register(registry);

        this.missLookupTimer = Timer.builder(PREFIX + ".lookup.duration")
                .tags(tags.and("result", "miss"))
                .description("Time spent in cache lookup resulting in a miss")
                .register(registry);

        // Rolling hit-rate gauge (supplier reads the ring buffer)
        Gauge.builder(PREFIX + ".hit_rate", this, PromptCacheMetrics::rollingHitRate)
                .tags(tags)
                .description("Rolling 5-minute prompt-cache hit rate (0.0–1.0)")
                .register(registry);

        LOG.infof("[PromptCacheMetrics] registered all meters for strategy=%s", strategy);
    }

    // -------------------------------------------------------------------------
    // Recording API — called from store implementations and plugins
    // -------------------------------------------------------------------------

    public void recordHit(int savedTokens) {
        hitsCounter.increment();
        savedTokensCounter.increment(savedTokens);
        savedTokensSummary.record(savedTokens);
        tickWindow(true);
    }

    public void recordMiss() {
        missesCounter.increment();
        tickWindow(false);
    }

    public void recordStore(int tokenCount) {
        storesCounter.increment();
    }

    public void recordEviction() {
        evictionsCounter.increment();
    }

    /**
     * Time a cache lookup and record the result.
     * Usage:
     * <pre>{@code
     * try (var sample = metrics.startLookupTimer()) {
     *     var result = store.lookup(hash);
     *     sample.stop(result.isPresent());
     *     return result;
     * }
     * }</pre>
     */
    public LookupSample startLookupTimer() {
        return new LookupSample(
                Timer.start(),
                hitLookupTimer,
                missLookupTimer
        );
    }

    /** Snapshot of current hit/miss counts for health checks. */
    public double currentHitRate() {
        double total = hitsCounter.count() + missesCounter.count();
        return total == 0 ? 0.0 : hitsCounter.count() / total;
    }

    // -------------------------------------------------------------------------
    // Rolling window
    // -------------------------------------------------------------------------

    private void tickWindow(boolean isHit) {
        long nowSec = System.currentTimeMillis() / 1000;
        long prev   = lastSecond.get();

        if (nowSec != prev) {
            // Advance ring buffer, zeroing stale buckets
            if (lastSecond.compareAndSet(prev, nowSec)) {
                long gap = Math.min(nowSec - prev, WINDOW_SECONDS);
                for (long s = 0; s < gap; s++) {
                    int idx = (int) ((prev + 1 + s) % WINDOW_SECONDS);
                    hitWindow[idx]   = 0;
                    totalWindow[idx] = 0;
                }
            }
        }

        int slot = (int) (nowSec % WINDOW_SECONDS);
        totalWindow[slot]++;
        if (isHit) hitWindow[slot]++;
    }

    private double rollingHitRate() {
        long sumHits = 0, sumTotal = 0;
        for (int i = 0; i < WINDOW_SECONDS; i++) {
            sumHits  += hitWindow[i];
            sumTotal += totalWindow[i];
        }
        return sumTotal == 0 ? 0.0 : (double) sumHits / sumTotal;
    }

    // -------------------------------------------------------------------------
    // Timer sample helper
    // -------------------------------------------------------------------------

    public static final class LookupSample implements AutoCloseable {
        private final Timer.Sample sample;
        private final Timer        hitTimer;
        private final Timer        missTimer;
        private boolean            hit = false;

        LookupSample(Timer.Sample sample, Timer hitTimer, Timer missTimer) {
            this.sample   = sample;
            this.hitTimer = hitTimer;
            this.missTimer = missTimer;
        }

        public void stop(boolean isHit) {
            this.hit = isHit;
        }

        @Override
        public void close() {
            sample.stop(hit ? hitTimer : missTimer);
        }
    }
}
