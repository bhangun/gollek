package tech.kayys.gollek.engine.inference;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.batch.BatchConfig;
import tech.kayys.gollek.spi.batch.BatchScheduler;
import tech.kayys.gollek.spi.batch.BatchStrategy;
import tech.kayys.gollek.spi.inference.InferenceEngine;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.time.Duration;

/**
 * CDI producer for {@link BatchScheduler}.
 *
 * <p>
 * Reads configuration from {@code application.properties}:
 * 
 * <pre>
 * gollek.batching.strategy=DYNAMIC          # STATIC | DYNAMIC | CONTINUOUS
 * gollek.batching.max-batch-size=8
 * gollek.batching.max-wait-ms=50
 * gollek.batching.max-concurrent-batches=4
 * gollek.batching.prefill-batch-size=4
 * gollek.batching.small-prompt-threshold=128
 * gollek.batching.enable-disaggregation=false
 * </pre>
 *
 * <p>
 * Falls back to {@link BatchConfig#defaultDynamic()} if the strategy value is
 * invalid.
 */
@ApplicationScoped
public class BatchSchedulerProducer {

    private static final Logger LOG = Logger.getLogger(BatchSchedulerProducer.class);

    @Inject
    InferenceEngine orchestrator;

    @ConfigProperty(name = "gollek.batching.strategy", defaultValue = "DYNAMIC")
    String strategy;

    @ConfigProperty(name = "gollek.batching.max-batch-size", defaultValue = "8")
    int maxBatchSize;

    @ConfigProperty(name = "gollek.batching.max-wait-ms", defaultValue = "50")
    long maxWaitMs;

    @ConfigProperty(name = "gollek.batching.max-concurrent-batches", defaultValue = "4")
    int maxConcurrentBatches;

    @ConfigProperty(name = "gollek.batching.prefill-batch-size", defaultValue = "4")
    int prefillBatchSize;

    @ConfigProperty(name = "gollek.batching.small-prompt-threshold", defaultValue = "128")
    int smallPromptThreshold;

    @ConfigProperty(name = "gollek.batching.enable-disaggregation", defaultValue = "false")
    boolean enableDisaggregation;

    @Produces
    @ApplicationScoped
    public BatchScheduler produceBatchScheduler() {
        BatchConfig cfg = buildConfig();
        LOG.infof("Producing BatchScheduler: strategy=%s, maxBatch=%d, maxWaitMs=%d",
                cfg.strategy(), cfg.maxBatchSize(), cfg.maxWaitTime().toMillis());
        return new DefaultBatchScheduler(orchestrator, cfg);
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private BatchConfig buildConfig() {
        BatchStrategy batchStrategy = parseStrategy(strategy);
        try {
            return new BatchConfig(
                    batchStrategy,
                    maxBatchSize,
                    Duration.ofMillis(maxWaitMs),
                    maxConcurrentBatches,
                    prefillBatchSize,
                    smallPromptThreshold,
                    enableDisaggregation);
        } catch (Exception e) {
            LOG.warnf("Invalid batch config (%s) — falling back to defaultDynamic()", e.getMessage());
            return BatchConfig.defaultDynamic();
        }
    }

    private BatchStrategy parseStrategy(String value) {
        try {
            return BatchStrategy.valueOf(value.toUpperCase());
        } catch (IllegalArgumentException e) {
            LOG.warnf("Unknown batching strategy '%s', defaulting to DYNAMIC", value);
            return BatchStrategy.DYNAMIC;
        }
    }
}
