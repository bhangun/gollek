package tech.kayys.gollek.inference.libtorch.sampling;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.LibTorchExecutionHints;
import tech.kayys.gollek.inference.libtorch.LibTorchFp8RowwiseTransformer;
import tech.kayys.gollek.inference.libtorch.LibTorchMetrics;
import tech.kayys.gollek.inference.libtorch.MixedPrecisionManager;
import tech.kayys.gollek.inference.libtorch.config.LibTorchProviderConfig;
import tech.kayys.gollek.inference.libtorch.LibTorchSessionManager;
import tech.kayys.gollek.inference.libtorch.TorchScriptRunner;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.spi.observability.AdapterSpec;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
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

    @Inject
    LibTorchMetrics metrics;

    @Inject
    MixedPrecisionManager mixedPrecisionManager;

    private final LibTorchFp8RowwiseTransformer rowwiseTransformer = new LibTorchFp8RowwiseTransformer();
    private final AtomicBoolean hybridFallbackForced = new AtomicBoolean(false);

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
        return generate(tenantId, modelId, modelPath, promptTokens, strategy, maxNewTokens, onToken, null,
                LibTorchExecutionHints.baseline());
    }

    public List<Long> generate(
            String tenantId, String modelId, Path modelPath,
            long[] promptTokens, SamplingStrategy strategy,
            int maxNewTokens, Consumer<Long> onToken,
            AdapterSpec adapterSpec) {
        return generate(tenantId, modelId, modelPath, promptTokens, strategy, maxNewTokens, onToken, adapterSpec,
                LibTorchExecutionHints.baseline());
    }

    public List<Long> generate(
            String tenantId, String modelId, Path modelPath,
            long[] promptTokens, SamplingStrategy strategy,
            int maxNewTokens, Consumer<Long> onToken,
            AdapterSpec adapterSpec,
            LibTorchExecutionHints executionHints) {

        List<Long> generated = new ArrayList<>();

        // Build full context (prompt + generated so far)
        List<Long> context = new ArrayList<>();
        for (long id : promptTokens) {
            context.add(id);
        }

        LibTorchSessionManager.SessionContext session = sessionManager.getSession(tenantId, modelId, config,
                adapterSpec);
        try {
            TorchScriptRunner runner = session.runner();

            for (int step = 0; step < maxNewTokens; step++) {
                long[] inputData = context.stream().mapToLong(l -> l).toArray();
                long[] shape = { 1, inputData.length };

                try (TorchTensor input = TorchTensor.fromLongArray(inputData, shape);
                        TorchTensor logits = forwardWithHybridFallback(runner, input, executionHints)) {

                    // Extract logits for the last token position
                    // logits shape: [1, seq_len, vocab_size] or [1, vocab_size]
                    float[] allLogits = logits.toFloatArray();
                    long[] logitsShape = logits.shape();

                    float[] lastTokenLogits;
                    int vocabSize;
                    if (logitsShape.length == 3) {
                        // [1, seq_len, vocab_size] → take last seq position
                        vocabSize = (int) logitsShape[2];
                        int seqLen = (int) logitsShape[1];
                        int offset = (seqLen - 1) * vocabSize;
                        lastTokenLogits = new float[vocabSize];
                        System.arraycopy(allLogits, offset, lastTokenLogits, 0, vocabSize);
                    } else {
                        // [1, vocab_size] or [vocab_size]
                        lastTokenLogits = allLogits;
                        vocabSize = lastTokenLogits.length;
                    }

                    if (executionHints != null && executionHints.fp8RowwiseEnabled()) {
                        if (!rowwiseTransformer.canApply(vocabSize, executionHints.fp8RowwiseScales())) {
                            throw new RuntimeException("FP8 rowwise scale mismatch at logits apply step");
                        }
                        rowwiseTransformer.applyInPlace(lastTokenLogits, executionHints.fp8RowwiseScales());
                    }

                    // Sample next token using the strategy
                    try (TorchTensor logitsTensor = TorchTensor.fromFloatArray(
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
                        long eosToken = Long.getLong("gollek.generator.eos-token", 2L);
                        if (nextToken == eosToken) { // Configurable EOS token
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
            sessionManager.releaseSession(tenantId, modelId, session, adapterSpec);
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
        return generate(tenantId, modelId, modelPath, promptTokens, strategy, maxNewTokens, null, null,
                LibTorchExecutionHints.baseline());
    }

    private TorchTensor forwardWithHybridFallback(
            TorchScriptRunner runner,
            TorchTensor input,
            LibTorchExecutionHints executionHints) {
        boolean hybridAttempt = executionHints != null
                && executionHints.hybridFp8Bf16AttentionEnabled()
                && !hybridFallbackForced.get();

        if (!hybridAttempt) {
            return runner.forward(input);
        }

        metrics.recordAdvancedHybridAttempt();
        TorchTensor mixedInput = null;
        TorchTensor mixedOutput = null;
        try {
            mixedInput = mixedPrecisionManager.castForHybridInput(input);
            mixedOutput = runner.forward(mixedInput);
            if (executionHints.fp8RowwiseEnabled()) {
                int expected = executionHints.fp8RowwiseScaleCount();
                int actual = inferVocabSize(mixedOutput);
                if (expected <= 0 || actual <= 0 || expected != actual) {
                    throw new RuntimeException("FP8 rowwise calibration mismatch: expected_scales="
                            + expected + ", actual_vocab=" + actual);
                }
            }
            TorchTensor logits = mixedPrecisionManager.castToFP32(mixedOutput);
            if (logits == mixedOutput) {
                mixedOutput = null;
            }
            metrics.recordAdvancedHybridSuccess();
            return logits;
        } catch (RuntimeException e) {
            if (hybridFallbackForced.compareAndSet(false, true)) {
                log.warnf(e, "Hybrid FP8/BF16 path failed once; forcing baseline fallback for subsequent requests");
            }
            metrics.recordAdvancedHybridFallback();
            return runner.forward(input);
        } finally {
            if (mixedOutput != null) {
                try {
                    mixedOutput.close();
                } catch (Exception ignored) {
                    // best-effort cleanup
                }
            }
            if (mixedInput != null && mixedInput != input) {
                try {
                    mixedInput.close();
                } catch (Exception ignored) {
                    // best-effort cleanup
                }
            }
        }
    }

    private int inferVocabSize(TorchTensor tensor) {
        long[] shape = tensor.shape();
        if (shape == null || shape.length == 0) {
            return -1;
        }
        if (shape.length >= 3) {
            return (int) shape[shape.length - 1];
        }
        if (shape.length == 2) {
            return (int) shape[1];
        }
        return (int) shape[0];
    }
}
