package tech.kayys.gollek.runtime.inference.batch;

import tech.kayys.gollek.runtime.inference.kv.KVCache;
import tech.kayys.gollek.runtime.inference.streaming.TokenStreamer;
import tech.kayys.gollek.spi.inference.Priority;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * A low-level runtime scheduling unit for the continuous batching scheduler.
 * <p>
 * This is NOT the same as {@code tech.kayys.gollek.spi.inference.InferenceRequest},
 * which is the high-level API request (messages, model, tools). This class
 * represents a single decode-stage request with its own token history,
 * KV cache, and streaming callback.
 * <p>
 * Named {@code BatchRequest} to avoid confusion with the SPI-level
 * {@code InferenceRequest}.
 */
public final class BatchRequest implements Comparable<BatchRequest> {

    /** Unique request ID. */
    public final UUID id;

    /** Tenant ID for multi-tenant scheduling. */
    public final String tenantId;

    /** Accumulated token IDs (prompt + generated). */
    public final List<Integer> tokens;

    /** Per-request KV cache. */
    public final KVCache cache;

    /** Streaming callback for generated tokens. */
    public final TokenStreamer streamer;

    /** Maximum tokens to generate. */
    public final int maxTokens;

    /** Priority level. */
    public final Priority priority;

    /** Time when the request was enqueued (nanos). */
    public final long enqueuedAt;

    /** Whether this request has finished generating. */
    public volatile boolean finished = false;

    /** Number of tokens generated so far. */
    public int generatedCount = 0;

    public BatchRequest(
        String tenantId,
        List<Integer> tokens,
        KVCache cache,
        TokenStreamer streamer,
        int maxTokens,
        Priority priority,
        long enqueuedAt
    ) {
        this.id = UUID.randomUUID();
        this.tenantId = tenantId;
        this.tokens = new ArrayList<>(tokens);
        this.cache = cache;
        this.streamer = streamer;
        this.maxTokens = maxTokens;
        this.priority = priority != null ? priority : Priority.NORMAL;
        this.enqueuedAt = enqueuedAt;
    }

    public BatchRequest(
        String tenantId,
        List<Integer> tokens,
        KVCache cache,
        TokenStreamer streamer,
        int maxTokens,
        Priority priority
    ) {
        this(tenantId, tokens, cache, streamer, maxTokens, priority, System.nanoTime());
    }

    public BatchRequest(
        String tenantId,
        List<Integer> tokens,
        KVCache cache,
        TokenStreamer streamer,
        int maxTokens
    ) {
        this(tenantId, tokens, cache, streamer, maxTokens, Priority.NORMAL, System.nanoTime());
    }

    /** Convenience constructor without tenant (single-tenant mode). */
    public BatchRequest(List<Integer> tokens, KVCache cache,
                        TokenStreamer streamer, int maxTokens) {
        this("default", tokens, cache, streamer, maxTokens, Priority.NORMAL, System.nanoTime());
    }

    @Override
    public int compareTo(BatchRequest other) {
        int priorityCmp = Integer.compare(this.priority.level(), other.priority.level());
        if (priorityCmp != 0) {
            return priorityCmp;
        }
        return Long.compare(this.enqueuedAt, other.enqueuedAt);
    }

    @Override
    public String toString() {
        return "BatchRequest[id=" + id
            + ", tenant=" + tenantId
            + ", tokens=" + tokens.size()
            + ", priority=" + priority
            + ", finished=" + finished + "]";
    }
}
