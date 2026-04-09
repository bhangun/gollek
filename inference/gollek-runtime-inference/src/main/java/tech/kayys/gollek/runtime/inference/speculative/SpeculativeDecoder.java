package tech.kayys.gollek.runtime.inference.speculative;

import tech.kayys.gollek.runtime.inference.kv.KVCache;
import tech.kayys.gollek.runtime.inference.streaming.TokenStreamer;

import java.util.ArrayList;
import java.util.List;

/**
 * Speculative decoding engine for 2x–5x inference speedup.
 * <p>
 * Uses two models:
 * <ul>
 *   <li><b>Draft model</b> (small, fast) — generates candidate tokens</li>
 *   <li><b>Target model</b> (large, accurate) — verifies candidates in one pass</li>
 * </ul>
 * <p>
 * The draft model proposes N tokens speculatively. The target model verifies
 * them all in a single forward pass. Accepted tokens are emitted immediately;
 * rejected tokens trigger a fallback to the target model.
 * <p>
 * This dramatically reduces the number of expensive target model forward passes.
 */
public final class SpeculativeDecoder {

    private final SpeculativeModel draftModel;
    private final SpeculativeModel targetModel;
    private final int draftSteps;

    /**
     * @param draftModel  small, fast model for speculation
     * @param targetModel large, accurate model for verification
     * @param draftSteps  number of speculative tokens per step (typically 4–8)
     */
    public SpeculativeDecoder(SpeculativeModel draftModel,
                               SpeculativeModel targetModel,
                               int draftSteps) {
        this.draftModel = draftModel;
        this.targetModel = targetModel;
        this.draftSteps = draftSteps;
    }

    /**
     * Execute one speculative decode step.
     *
     * @param tokens current token sequence
     * @param cache  KV cache for the target model
     * @return list of accepted tokens (may be 0 to draftSteps)
     */
    public List<Integer> decodeStep(List<Integer> tokens, KVCache cache) {
        // 1. Draft: generate candidate tokens
        List<Integer> draftTokens = new ArrayList<>();
        List<Integer> working = new ArrayList<>(tokens);

        for (int i = 0; i < draftSteps; i++) {
            int t = draftModel.nextToken(working);
            working.add(t);
            draftTokens.add(t);
        }

        // 2. Verify: check all drafts in one target model pass
        int[] verified = targetModel.verifyBatch(tokens, draftTokens, cache);

        // 3. Accept: keep longest valid prefix
        List<Integer> accepted = new ArrayList<>();
        for (int i = 0; i < draftTokens.size(); i++) {
            if (i < verified.length && verified[i] == draftTokens.get(i)) {
                accepted.add(draftTokens.get(i));
            } else {
                break;
            }
        }

        return accepted;
    }

    /**
     * Run full speculative generation with streaming.
     *
     * @param tokens    initial token sequence
     * @param cache     KV cache
     * @param maxTokens maximum tokens to generate
     * @param eosToken  end-of-sequence token ID
     * @param streamer  streaming callback
     */
    public void generate(
        List<Integer> tokens,
        KVCache cache,
        int maxTokens,
        int eosToken,
        TokenStreamer streamer
    ) {
        List<Integer> working = new ArrayList<>(tokens);
        int generated = 0;

        try {
            while (generated < maxTokens) {
                List<Integer> accepted = decodeStep(working, cache);

                if (accepted.isEmpty()) {
                    // Fallback: single target model step
                    int t = targetModel.nextToken(working);
                    working.add(t);
                    streamer.onToken(t, "");
                    generated++;

                    if (t == eosToken) break;
                } else {
                    for (int t : accepted) {
                        working.add(t);
                        streamer.onToken(t, "");
                        generated++;

                        if (t == eosToken) {
                            streamer.onComplete();
                            return;
                        }
                    }
                }
            }

            streamer.onComplete();

        } catch (Exception e) {
            streamer.onError(e);
        }
    }

    /**
     * Model interface for speculative decoding.
     * Both draft and target models must implement this.
     */
    public interface SpeculativeModel {

        /** Generate the next token given the sequence so far. */
        int nextToken(List<Integer> tokens);

        /**
         * Verify a batch of draft tokens against the target model.
         * Returns the tokens that the target model would have produced
         * at each position (for comparison with draft tokens).
         */
        int[] verifyBatch(List<Integer> prefix, List<Integer> draftTokens, KVCache cache);
    }
}
