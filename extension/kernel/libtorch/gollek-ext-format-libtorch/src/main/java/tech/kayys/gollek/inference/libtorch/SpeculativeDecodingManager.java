package tech.kayys.gollek.inference.libtorch;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.Tensor;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Manages speculative decoding using a Draft Model and a Target Model.
 * <p>
 * basic implementation:
 * 1. Draft model generates K tokens.
 * 2. Target model verifies K tokens in parallel.
 * 3. Accepted tokens are returned.
 */
@ApplicationScoped
public class SpeculativeDecodingManager {

    private static final Logger LOG = Logger.getLogger(SpeculativeDecodingManager.class);

    @Inject
    LibTorchSessionManager sessionManager;

    @Inject
    LibTorchProviderConfig config;

    // Configuration for speculative decoding
    private static final int DEFAULT_LOOKAHEAD = 5;

    /**
     * Executes a speculative decoding step.
     *
     * @param tenantId        Tenant ID
     * @param draftModelId    Draft model ID
     * @param draftModelPath  Path to draft model
     * @param targetModelId   Target model ID
     * @param targetModelPath Path to target model
     * @param promptIds       Input token IDs
     * @return Accepted token IDs
     */
    public int[] executeSpeculativeStep(
            String tenantId,
            String draftModelId, Path draftModelPath,
            String targetModelId, Path targetModelPath,
            long[] promptIds) {

        // 1. Generate K draft tokens
        long[] draftTokens = generateDraftTokens(tenantId, draftModelId, draftModelPath, promptIds, DEFAULT_LOOKAHEAD);

        // 2. Verify with Target Model
        return verifyTokens(tenantId, targetModelId, targetModelPath, promptIds, draftTokens);

    }

    private long[] generateDraftTokens(String tenantId, String modelId, Path modelPath, long[] promptIds, int k) {
        List<Long> currentIds = new ArrayList<>();
        for (long id : promptIds) {
            currentIds.add(id);
        }

        long[] drafts = new long[k];

        // Use session manager to acquire a runner
        LibTorchSessionManager.SessionContext ctx = sessionManager.getSession(tenantId, modelId, config);
        try {
            TorchScriptRunner runner = ctx.runner();
            for (int i = 0; i < k; i++) {
                long[] inputData = currentIds.stream().mapToLong(l -> l).toArray();
                long[] shape = { 1, inputData.length }; // Batch size 1

                try (Tensor input = Tensor.fromLongArray(inputData, shape);
                        Tensor logits = runner.forward(input);
                        // Simpler: logits.argmax(-1) gives [batch, seq]. Take last element.
                        Tensor predictions = logits.argmax(-1)) {

                    long[] predArray = predictions.toLongArray();
                    long nextToken = predArray[predArray.length - 1];

                    drafts[i] = nextToken;
                    currentIds.add(nextToken);
                }
            }
        } catch (Exception e) {
            LOG.errorf(e, "Error generating draft tokens with model %s", modelId);
            throw new RuntimeException("Draft generation failed", e);
        } finally {
            sessionManager.releaseSession(tenantId, modelId, ctx);
        }

        return drafts;
    }

    @Inject
    ContinuousBatchingManager batchingManager;

    private int[] verifyTokens(String tenantId, String modelId, Path modelPath, long[] promptIds, long[] draftTokens) {
        // Prepare input: prompt + draft
        long[] fullInput = new long[promptIds.length + draftTokens.length];
        System.arraycopy(promptIds, 0, fullInput, 0, promptIds.length);
        System.arraycopy(draftTokens, 0, fullInput, promptIds.length, draftTokens.length);

        try {
            // Create input tensor for batch request
            // Note: ContinuousBatchingManager expects float[] input for now??
            // Wait, LibTorchProvider.inferBatched converts to float array.
            // But ContinuousBatchingManager.BatchRequest expects a Tensor input.
            // LibTorchProvider creates Tensor from Float array.
            // If our model is Long (embedding layer), we need Long tensor.
            // But LibTorchProvider assumes Float input!
            // ContinuousBatchingManager takes `Tensor input`.
            // So we can pass a Long tensor!

            // Create Float Tensor to match LibTorchProvider behavior (avoid mixed types in
            // batch)
            float[] floatInput = new float[fullInput.length];
            for (int i = 0; i < fullInput.length; i++) {
                floatInput[i] = (float) fullInput[i];
            }
            Tensor input = Tensor.fromFloatArray(floatInput, new long[] { 1, fullInput.length });

            // Define handler for verification
            java.util.function.Function<Tensor, InferenceResponse> handler = tensor -> {
                // Tensor shape: [1, seq_len, vocab] or [1, seq_len] if argmaxed?
                // No, runner.forward returns logits/probabilities.
                // We need to argmax here.

                // tensor is the slice for this request.
                try (Tensor argmax = tensor.argmax(-1)) {
                    long[] predArray = argmax.toLongArray();

                    List<Integer> accepted = new ArrayList<>();
                    // Verify loop
                    for (int i = 0; i < draftTokens.length; i++) {
                        int verifyPos = promptIds.length + i - 1;
                        if (verifyPos >= predArray.length)
                            break;
                        long predictedToken = predArray[verifyPos];

                        if (predictedToken == draftTokens[i]) {
                            accepted.add((int) predictedToken);
                        } else {
                            accepted.add((int) predictedToken); // Bonus token
                            break;
                        }
                    }

                    int[] acceptedArr = accepted.stream().mapToInt(Integer::intValue).toArray();
                    return InferenceResponse.builder()
                            .requestId("verification")
                            .content("")
                            .tokenIds(acceptedArr)
                            .build();
                }
            };

            CompletableFuture<InferenceResponse> future = new CompletableFuture<>();
            batchingManager.enqueue(new ContinuousBatchingManager.BatchRequest(
                    tenantId, modelId, modelPath, "verification-" + System.nanoTime(),
                    input, handler, future));

            // Wait for result
            InferenceResponse response = future.join();
            return response.getTokenIds();

        } catch (Exception e) {
            LOG.errorf(e, "Error verifying tokens with model %s", modelId);
            throw new RuntimeException("Verification failed", e);
        }
    }
}
