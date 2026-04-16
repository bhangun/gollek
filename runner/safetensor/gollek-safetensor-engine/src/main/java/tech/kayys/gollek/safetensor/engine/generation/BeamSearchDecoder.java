/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * BeamSearchDecoder.java
 * ───────────────────────
 * Beam search decoder for the direct inference backend.
 *
 * What beam search is
 * ════════════════════
 * Greedy decoding always picks the single highest-probability next token.
 * This can lead to locally-optimal but globally-suboptimal sequences.
 *
 * Beam search maintains K candidate sequences (beams) simultaneously,
 * expanding each by the top-K next tokens, then keeping only the K beams
 * with the highest cumulative log-probability.  The final output is the
 * beam with the highest normalised score.
 *
 * When to use beam search
 * ═══════════════════════
 * - Machine translation (high beam width = 4-8 common)
 * - Summarisation tasks
 * - Any task where output quality matters more than speed
 *
 * When NOT to use
 * ═══════════════
 * - Chat / conversational tasks (greedy or top-p is better)
 * - Creative writing (introduces repetition artefacts)
 * - Streaming (can't stream K beams easily)
 *
 * Length normalisation
 * ════════════════════
 * Without normalisation, shorter sequences score higher (fewer -log p terms).
 * The normalised score is:
 *
 *   score = cumulative_logprob / (seqLen ^ α)
 *
 * where α is the length_penalty (typically 0.6–1.0).
 * α = 1.0 → fully normalise.
 * α = 0.0 → no normalisation (equivalent to greedy).
 *
 * Configuration
 * ═════════════
 * Via GenerationConfig.builder().strategy(BEAM).beamWidth(4).build()
 * Length penalty defaults to 0.7.
 */
package tech.kayys.gollek.safetensor.engine.generation;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;

import java.util.*;

/**
 * Beam search sequence decoder.
 *
 * <p>
 * Activated when {@link GenerationConfig#isBeamSearch()} is true.
 * Inject and call {@link #decode} from {@code DirectInferenceEngine.runLoop}.
 */
@ApplicationScoped
public class BeamSearchDecoder {

    private static final Logger log = Logger.getLogger(BeamSearchDecoder.class);

    private static final float LENGTH_PENALTY = 0.7f;

    @Inject
    DirectForwardPass forwardPass;
    @Inject
    KVCacheManager kvCacheManager;

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Run beam search decoding.
     *
     * @param prefixIds    prompt token IDs (already prefilled)
     * @param prefixLogits logits from the last prefill step
     * @param weights      model weight map
     * @param config       model config
     * @param arch         architecture
     * @param genCfg       generation config (beamWidth, maxNewTokens, stopTokenIds)
     * @param tokenizer    for EOS detection
     * @return the best decoded sequence as a token ID array
     */
    public int[] decode(int[] prefixIds,
            float[] prefixLogits,
            Map<String, TorchTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            GenerationConfig genCfg,
            Tokenizer tokenizer) {

        int beamWidth = genCfg.beamWidth();
        int maxNew = genCfg.maxNewTokens();
        int eosId = tokenizer.eosTokenId();
        Set<Integer> stops = new HashSet<>(genCfg.stopTokenIds());
        stops.add(eosId);

        log.debugf("BeamSearch: beamWidth=%d maxNew=%d", beamWidth, maxNew);

        // ── Initialise beams from top-K of initial logits ────────────────────
        float[] logprobs = logSoftmax(prefixLogits);
        List<Beam> beams = topKBeams(logprobs, beamWidth);

        // ── Expand beams step by step ────────────────────────────────────────
        List<Beam> finishedBeams = new ArrayList<>();

        for (int step = 0; step < maxNew && !beams.isEmpty(); step++) {
            List<Beam> candidates = new ArrayList<>(beams.size() * beamWidth);

            for (Beam beam : beams) {
                if (beam.finished()) {
                    finishedBeams.add(beam);
                    continue;
                }

                // Run one decode step for this beam's last token
                float[] stepLogits = runDecodeStep(
                        beam.lastTokenId(), prefixIds.length + step,
                        weights, config, arch, beam.kvSession());

                float[] stepLogprobs = logSoftmax(stepLogits);
                int[] topK = topKIndices(stepLogprobs, beamWidth);

                for (int tok : topK) {
                    float newScore = beam.cumulativeLogprob() + stepLogprobs[tok];
                    List<Integer> newSeq = new ArrayList<>(beam.tokenIds());
                    newSeq.add(tok);
                    boolean finished = stops.contains(tok) || step + 1 >= maxNew;
                    candidates.add(new Beam(newSeq, newScore, beam.kvSession(), finished));
                }
            }

            // Keep top-beamWidth candidates by normalised score
            candidates.sort(Comparator.comparingDouble(b -> -b.normalisedScore(LENGTH_PENALTY)));

            beams = candidates.stream()
                    .limit(beamWidth)
                    .filter(b -> !b.finished())
                    .toList();

            candidates.stream().filter(Beam::finished).forEach(finishedBeams::add);
        }

        // ── Select best finished beam ─────────────────────────────────────────
        List<Beam> allComplete = new ArrayList<>(finishedBeams);
        allComplete.addAll(beams); // add any still-running beams as fallback

        Beam best = allComplete.stream()
                .max(Comparator.comparingDouble(b -> b.normalisedScore(LENGTH_PENALTY)))
                .orElse(beams.isEmpty() ? null : beams.get(0));

        if (best == null)
            return new int[0];
        return best.tokenIds().stream().mapToInt(Integer::intValue).toArray();
    }

    // ─────────────────────────────────────────────────────────────────────────

    private float[] runDecodeStep(int tokenId, int pos,
            Map<String, TorchTensor> weights,
            ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kv) {
        try {
            return forwardPass.decode(tokenId, pos, weights, config, arch, kv);
        } catch (Exception e) {
            log.warnf(e, "BeamSearch: decode step failed at pos=%d", pos);
            return new float[config.vocabSize()];
        }
    }

    private List<Beam> topKBeams(float[] logprobs, int k) {
        int[] topK = topKIndices(logprobs, k);
        List<Beam> beams = new ArrayList<>(k);
        for (int tok : topK) {
            beams.add(new Beam(List.of(tok), logprobs[tok], null, false));
        }
        return beams;
    }

    private static int[] topKIndices(float[] arr, int k) {
        k = Math.min(k, arr.length);
        Integer[] idx = new Integer[arr.length];
        for (int i = 0; i < idx.length; i++)
            idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(arr[b], arr[a]));
        int[] result = new int[k];
        for (int i = 0; i < k; i++)
            result[i] = idx[i];
        return result;
    }

    private static float[] logSoftmax(float[] logits) {
        float max = logits[0];
        for (float v : logits)
            if (v > max)
                max = v;
        double sum = 0;
        for (float v : logits)
            sum += Math.exp(v - max);
        float lse = (float) Math.log(sum);
        float[] out = new float[logits.length];
        for (int i = 0; i < logits.length; i++)
            out[i] = logits[i] - max - lse;
        return out;
    }

    // ── Beam value type ───────────────────────────────────────────────────────

    private record Beam(
            List<Integer> tokenIds,
            float cumulativeLogprob,
            KVCacheManager.KVCacheSession kvSession,
            boolean finished) {

        int lastTokenId() {
            return tokenIds.isEmpty() ? -1 : tokenIds.get(tokenIds.size() - 1);
        }

        /** Length-normalised score for beam comparison. */
        double normalisedScore(float alpha) {
            if (tokenIds.isEmpty())
                return Double.NEGATIVE_INFINITY;
            return cumulativeLogprob / Math.pow(tokenIds.size(), alpha);
        }
    }
}
