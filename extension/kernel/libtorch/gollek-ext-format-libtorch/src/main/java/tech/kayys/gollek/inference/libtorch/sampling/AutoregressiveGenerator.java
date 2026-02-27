package tech.kayys.gollek.inference.libtorch.sampling;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.LibTorchProviderConfig;
import tech.kayys.gollek.inference.libtorch.LibTorchSessionManager;
import tech.kayys.gollek.inference.libtorch.TorchScriptRunner;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Autoregressive token generator with pluggable sampling strategies.
 * <p>
 * Generates tokens one at a time using a TorchScript model and a
 * configurable {@link SamplingStrategy}. Supports both complete
 * generation (returns all tokens) and streaming (calls a callback
 * per token for SSE/WebSocket delivery).
 *
 * <h3>Usage:</h3>
 * 
 * <pre>{@code
 * var strategy = new TopPSampler(0.9, 0.8);
 * List<Long> tokens = generator.generate(
 *         tenantId, modelId, modelPath, promptIds,
 *         strategy, 256, token -> sendSSE(token));
 * }</pre>
 */
@ApplicationScoped
public class AutoregressiveGenerator {

    private static final Logger log = Logger.getLogger(AutoregressiveGenerator.class);

    @Inject
    LibTorchSessionManager sessionManager;

    @Inject
    LibTorchProviderConfig config;

    /**
     * Generate tokens autoregressively using the given sampling strategy.
     *
     * @param tenantId     tenant identifier
     * @param modelId      model identifier
     * @param modelPath    path to the model file
     * @param promptTokens input prompt token IDs
     * @param strategy     sampling strategy (greedy, top-k, top-p, etc.)
     * @param maxNewTokens maximum number of tokens to generate
     * @param onToken      callback invoked for each generated token (for
     *                     streaming);
     *                     may be null for non-streaming use
     * @return list of all generated token IDs (excluding prompt)
     */
    public List<Long> generate(
            String tenantId, String modelId, Path modelPath,
            long[] promptTokens, SamplingStrategy strategy,
            int maxNewTokens, Consumer<Long> onToken) {

        List<Long> generated = new ArrayList<>();

        // Build full context (prompt + generated so far)
        List<Long> context = new ArrayList<>();
        for (long id : promptTokens) {
            context.add(id);
        }

        LibTorchSessionManager.SessionContext session = sessionManager.getSession(tenantId, modelId, config);
        try {
            TorchScriptRunner runner = session.runner();

            for (int step = 0; step < maxNewTokens; step++) {
                long[] inputData = context.stream().mapToLong(l -> l).toArray();
                long[] shape = { 1, inputData.length };

                try (Tensor input = Tensor.fromLongArray(inputData, shape);
                        Tensor logits = runner.forward(input)) {

                    // Extract logits for the last token position
                    // logits shape: [1, seq_len, vocab_size] or [1, vocab_size]
                    float[] allLogits = logits.toFloatArray();
                    long[] logitsShape = logits.shape();

                    float[] lastTokenLogits;
                    if (logitsShape.length == 3) {
                        // [1, seq_len, vocab_size] → take last seq position
                        int vocabSize = (int) logitsShape[2];
                        int seqLen = (int) logitsShape[1];
                        int offset = (seqLen - 1) * vocabSize;
                        lastTokenLogits = new float[vocabSize];
                        System.arraycopy(allLogits, offset, lastTokenLogits, 0, vocabSize);
                    } else {
                        // [1, vocab_size] or [vocab_size]
                        lastTokenLogits = allLogits;
                    }

                    // Sample next token using the strategy
                    try (Tensor logitsTensor = Tensor.fromFloatArray(
                            lastTokenLogits, new long[] { lastTokenLogits.length })) {
                        long nextToken = strategy.sample(logitsTensor);

                        generated.add(nextToken);
                        context.add(nextToken);

                        // Streaming callback
                        if (onToken != null) {
                            onToken.accept(nextToken);
                        }

                        // EOS detection (configurable — token 0 or 2 are common EOS)
                        // TODO: Make EOS token configurable
                        if (nextToken == 2) { // Common EOS token
                            log.debugf("EOS token encountered at step %d", step);
                            break;
                        }
                    }
                }
            }

            log.debugf("Generated %d tokens using %s strategy", generated.size(), strategy.name());
        } catch (Exception e) {
            log.errorf(e, "Autoregressive generation failed at step %d", generated.size());
            throw new RuntimeException("Generation failed", e);
        } finally {
            sessionManager.releaseSession(tenantId, modelId, session);
        }

        return generated;
    }

    /**
     * Generate tokens without streaming callback.
     */
    public List<Long> generate(
            String tenantId, String modelId, Path modelPath,
            long[] promptTokens, SamplingStrategy strategy,
            int maxNewTokens) {
        return generate(tenantId, modelId, modelPath, promptTokens, strategy, maxNewTokens, null);
    }
}
