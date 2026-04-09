package tech.kayys.gollek.sdk.session;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.optim.Optimizer;
import tech.kayys.gollek.sdk.GollekClient;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;

/**
 * Inference session — manages a stateful inference context with metrics tracking,
 * KV cache lifecycle, and request batching.
 *
 * <p>Equivalent to a scoped inference context in TensorFlow Serving or
 * a session in ONNX Runtime.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * try (InferenceSession session = InferenceSession.create(client)) {
 *     GollekClient.GenerationResult r = session.run("Explain transformers");
 *     System.out.println(session.metrics());
 * }
 * }</pre>
 */
public final class InferenceSession implements AutoCloseable {

    private final GollekClient client;
    private final AtomicLong totalTokens   = new AtomicLong();
    private final AtomicLong totalRequests = new AtomicLong();
    private final AtomicLong totalMs       = new AtomicLong();

    /**
     * Session metrics snapshot.
     *
     * @param totalRequests number of requests processed
     * @param totalTokens   total tokens generated
     * @param avgLatencyMs  average latency per request in milliseconds
     * @param tokensPerSec  average throughput in tokens/second
     */
    public record SessionMetrics(long totalRequests, long totalTokens,
                                  double avgLatencyMs, double tokensPerSec) {}

    private InferenceSession(GollekClient client) { this.client = client; }

    /**
     * Creates a new inference session backed by the given client.
     *
     * @param client initialized {@link GollekClient}
     * @return new session
     */
    public static InferenceSession create(GollekClient client) {
        return new InferenceSession(client);
    }

    /**
     * Runs a single generation request and tracks metrics.
     *
     * @param prompt input prompt
     * @return generation result
     */
    public GollekClient.GenerationResult run(String prompt) {
        return run(GollekClient.GenerationRequest.of(prompt));
    }

    /**
     * Runs a generation request with full configuration.
     *
     * @param request generation request
     * @return generation result
     */
    public GollekClient.GenerationResult run(GollekClient.GenerationRequest request) {
        long start = System.currentTimeMillis();
        GollekClient.GenerationResult result = client.generate(request);
        long elapsed = System.currentTimeMillis() - start;
        totalRequests.incrementAndGet();
        totalTokens.addAndGet(result.tokenCount());
        totalMs.addAndGet(elapsed);
        return result;
    }

    /**
     * Runs a batch of prompts in parallel.
     *
     * @param prompts list of input prompts
     * @return list of results
     */
    public List<GollekClient.GenerationResult> runBatch(List<String> prompts) {
        return client.generateBatch(prompts);
    }

    /**
     * Returns current session metrics.
     *
     * @return {@link SessionMetrics} snapshot
     */
    public SessionMetrics metrics() {
        long reqs = totalRequests.get();
        long toks = totalTokens.get();
        long ms   = totalMs.get();
        double avgMs = reqs > 0 ? (double) ms / reqs : 0;
        double tps   = ms > 0 ? toks * 1000.0 / ms : 0;
        return new SessionMetrics(reqs, toks, avgMs, tps);
    }

    /** Checks if the backend supports a specific feature. */
    public boolean supports(GollekClient.Feature feature) { return client.supports(feature); }

    @Override public void close() { client.close(); }
}
