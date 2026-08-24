package tech.kayys.gollek.extension;

import org.jboss.logging.Logger;
import tech.kayys.gollek.runner.ModelRunner;
import tech.kayys.gollek.runner.RunnerStats;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.alkhawarizm.spi.model.ResourceMetrics;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Shared base for Aljabr ModelRunner implementations.
 *
 * <p>Tracks request counts and latency. Concrete subclasses implement
 * {@link ModelRunner#initialize}, {@link ModelRunner#infer}, {@link ModelRunner#health},
 * and {@link ModelRunner#close}.
 */
public abstract class AbstractGollekRunner implements ModelRunner {

    protected final Logger log = Logger.getLogger(getClass());

    protected final AtomicLong totalRequests = new AtomicLong();
    protected final AtomicLong totalFailures = new AtomicLong();
    protected final AtomicLong totalLatencyMs = new AtomicLong();

    protected volatile boolean initialized = false;

    @Override
    public RunnerStats metrics() {
        long reqs = totalRequests.get();
        long avg = reqs > 0 ? totalLatencyMs.get() / reqs : 0L;
        return RunnerStats.builder()
                .totalRequests(reqs)
                .failedRequests(totalFailures.get())
                .averageLatencyMs(avg)
                .p95LatencyMs(avg) // approximation; replace with histogram in production
                .p99LatencyMs(avg)
                .build();
    }

    @Override
    public ResourceMetrics getMetrics() {
        return new ResourceMetrics(
                0,
                0,
                0,
                0,
                (int) (totalRequests.get() - totalFailures.get()));
    }

    // ── Tokenizer / sampler stubs ────────────────────────────────────────────

    protected int[] tokenize(InferenceRequest request) {
        int charCount = request.getMessages().stream()
                .mapToInt(m -> m.getContent() != null ? m.getContent().length() : 0)
                .sum();
        return new int[Math.max(1, charCount / 4)];
    }

    protected String detokenize(int tokenId) {
        return tokenId == 0 ? "" : " token";
    }

    protected boolean isEos(int tokenId) {
        return tokenId == 2; // </s> for LLaMA-family
    }

    protected int sampleGreedy(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    protected int getMaxTokens(InferenceRequest request) {
        Object v = request.getParameters().get("max_tokens");
        if (v instanceof Number n) {
            return n.intValue();
        }
        return 2048;
    }
}
