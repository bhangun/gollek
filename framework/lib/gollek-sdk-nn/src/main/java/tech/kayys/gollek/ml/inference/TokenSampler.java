package tech.kayys.gollek.ml.inference;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.*;

/**
 * Token sampling strategies for autoregressive language model generation.
 *
 * <p>Provides greedy, top-k, top-p (nucleus), and temperature sampling,
 * plus beam search for higher-quality generation.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * // Greedy decoding
 * int token = TokenSampler.greedy(logits);
 *
 * // Top-p nucleus sampling
 * int token = TokenSampler.topP(logits, p=0.9f, temperature=0.8f);
 *
 * // Beam search
 * List<int[]> beams = TokenSampler.beamSearch(model, inputIds, beamWidth=4, maxLen=50);
 * }</pre>
 */
public final class TokenSampler {

    private TokenSampler() {}

    /**
     * Greedy decoding — returns the token with the highest logit.
     *
     * @param logits raw model output {@code [vocabSize]}
     * @return token index with maximum logit
     */
    public static int greedy(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /**
     * Temperature sampling — scales logits before softmax.
     *
     * <p>Temperature {@code T < 1} sharpens the distribution (more deterministic),
     * {@code T > 1} flattens it (more random).
     *
     * @param logits      raw model output {@code [vocabSize]}
     * @param temperature scaling factor (must be positive)
     * @return sampled token index
     */
    public static int temperature(float[] logits, float temperature) {
        float[] scaled = new float[logits.length];
        for (int i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
        return sampleFromProbs(softmax(scaled));
    }

    /**
     * Top-k sampling — restricts sampling to the k highest-probability tokens.
     *
     * @param logits      raw model output {@code [vocabSize]}
     * @param k           number of top tokens to consider
     * @param temperature temperature scaling (1.0 = no scaling)
     * @return sampled token index
     */
    public static int topK(float[] logits, int k, float temperature) {
        // Find top-k indices
        Integer[] idx = new Integer[logits.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(logits[b], logits[a]));

        float[] filtered = new float[logits.length];
        Arrays.fill(filtered, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < Math.min(k, logits.length); i++)
            filtered[idx[i]] = logits[idx[i]] / temperature;

        return sampleFromProbs(softmax(filtered));
    }

    /**
     * Top-p (nucleus) sampling — samples from the smallest set of tokens
     * whose cumulative probability exceeds {@code p}.
     *
     * @param logits      raw model output {@code [vocabSize]}
     * @param p           cumulative probability threshold (e.g. 0.9)
     * @param temperature temperature scaling
     * @return sampled token index
     */
    public static int topP(float[] logits, float p, float temperature) {
        float[] scaled = new float[logits.length];
        for (int i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
        float[] probs = softmax(scaled);

        // Sort by probability descending
        Integer[] idx = new Integer[probs.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(probs[b], probs[a]));

        // Keep tokens until cumulative prob >= p
        float[] filtered = new float[probs.length];
        float cumProb = 0f;
        for (int i = 0; i < idx.length; i++) {
            filtered[idx[i]] = probs[idx[i]];
            cumProb += probs[idx[i]];
            if (cumProb >= p) break;
        }
        // Renormalize
        float sum = 0; for (float v : filtered) sum += v;
        if (sum > 0) for (int i = 0; i < filtered.length; i++) filtered[i] /= sum;

        return sampleFromProbs(filtered);
    }

    /**
     * Beam search — maintains {@code beamWidth} candidate sequences and
     * returns the top beam at the end.
     *
     * @param logitsFn    function: token IDs → next-token logits {@code [vocabSize]}
     * @param inputIds    initial token IDs
     * @param beamWidth   number of beams to maintain
     * @param maxNewTokens maximum tokens to generate
     * @param eosId       end-of-sequence token ID (stops beam when generated)
     * @return best sequence of generated token IDs
     */
    public static int[] beamSearch(
            java.util.function.Function<int[], float[]> logitsFn,
            int[] inputIds, int beamWidth, int maxNewTokens, int eosId) {

        // Each beam: (score, token sequence)
        record Beam(float score, int[] tokens) {}

        List<Beam> beams = new ArrayList<>();
        beams.add(new Beam(0f, inputIds.clone()));

        for (int step = 0; step < maxNewTokens; step++) {
            List<Beam> candidates = new ArrayList<>();
            for (Beam beam : beams) {
                if (beam.tokens()[beam.tokens().length - 1] == eosId) {
                    candidates.add(beam); continue;
                }
                float[] logits = logitsFn.apply(beam.tokens());
                float[] probs  = softmax(logits);

                // Expand top-k candidates
                Integer[] idx = new Integer[probs.length];
                for (int i = 0; i < idx.length; i++) idx[i] = i;
                Arrays.sort(idx, (a, b) -> Float.compare(probs[b], probs[a]));

                for (int i = 0; i < Math.min(beamWidth, idx.length); i++) {
                    int tok = idx[i];
                    float score = beam.score() + (float) Math.log(Math.max(probs[tok], 1e-10f));
                    int[] newTokens = Arrays.copyOf(beam.tokens(), beam.tokens().length + 1);
                    newTokens[newTokens.length - 1] = tok;
                    candidates.add(new Beam(score, newTokens));
                }
            }
            // Keep top beamWidth candidates
            candidates.sort((a, b) -> Float.compare(b.score(), a.score()));
            beams = candidates.subList(0, Math.min(beamWidth, candidates.size()));
        }
        return beams.get(0).tokens();
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float[] probs = new float[logits.length];
        float sum = 0;
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] == Float.NEGATIVE_INFINITY) continue;
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        if (sum > 0) for (int i = 0; i < probs.length; i++) probs[i] /= sum;
        return probs;
    }

    private static int sampleFromProbs(float[] probs) {
        float r = new Random().nextFloat();
        float cum = 0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r <= cum) return i;
        }
        return probs.length - 1;
    }
}
