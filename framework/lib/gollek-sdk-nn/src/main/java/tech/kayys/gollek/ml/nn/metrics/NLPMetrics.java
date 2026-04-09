package tech.kayys.gollek.ml.nn.metrics;

import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.*;

/**
 * NLP evaluation metrics: BLEU score, Perplexity, and ROUGE-N.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * // BLEU
 * float bleu = NLPMetrics.bleu(hypothesis, references, maxN=4);
 *
 * // Perplexity from cross-entropy loss
 * float ppl = NLPMetrics.perplexity(avgCrossEntropyLoss);
 *
 * // ROUGE-1
 * float rouge1 = NLPMetrics.rouge(hypothesis, reference, n=1);
 * }</pre>
 */
public final class NLPMetrics {

    private NLPMetrics() {}

    /**
     * Computes corpus BLEU score (up to 4-gram).
     *
     * <p>Uses modified n-gram precision with brevity penalty.
     *
     * @param hypothesis tokenized hypothesis sentence
     * @param references list of tokenized reference sentences
     * @param maxN       maximum n-gram order (typically 4)
     * @return BLEU score in [0, 1]
     */
    public static float bleu(List<String> hypothesis, List<List<String>> references, int maxN) {
        if (hypothesis.isEmpty()) return 0f;

        float logSum = 0f;
        for (int n = 1; n <= maxN; n++) {
            float precision = ngramPrecision(hypothesis, references, n);
            if (precision == 0) return 0f;
            logSum += Math.log(precision);
        }

        // Brevity penalty
        int hypLen = hypothesis.size();
        int refLen = references.stream()
            .mapToInt(List::size)
            .min().orElse(hypLen);
        float bp = hypLen >= refLen ? 1f : (float) Math.exp(1 - (float) refLen / hypLen);

        return bp * (float) Math.exp(logSum / maxN);
    }

    /**
     * Computes perplexity from average cross-entropy loss.
     *
     * <p>{@code PPL = exp(avg_loss)}
     *
     * @param avgCrossEntropyLoss average negative log-likelihood per token
     * @return perplexity (lower is better; 1.0 = perfect)
     */
    public static float perplexity(float avgCrossEntropyLoss) {
        return (float) Math.exp(avgCrossEntropyLoss);
    }

    /**
     * Computes ROUGE-N: n-gram recall between hypothesis and reference.
     *
     * @param hypothesis tokenized hypothesis
     * @param reference  tokenized reference
     * @param n          n-gram order (1 or 2 typically)
     * @return ROUGE-N F1 score in [0, 1]
     */
    public static float rouge(List<String> hypothesis, List<String> reference, int n) {
        Map<String, Integer> hypNgrams = countNgrams(hypothesis, n);
        Map<String, Integer> refNgrams = countNgrams(reference, n);

        int overlap = 0;
        for (var entry : hypNgrams.entrySet())
            overlap += Math.min(entry.getValue(), refNgrams.getOrDefault(entry.getKey(), 0));

        int refTotal = refNgrams.values().stream().mapToInt(Integer::intValue).sum();
        int hypTotal = hypNgrams.values().stream().mapToInt(Integer::intValue).sum();

        float recall    = refTotal > 0 ? (float) overlap / refTotal : 0f;
        float precision = hypTotal > 0 ? (float) overlap / hypTotal : 0f;
        return (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0f;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static float ngramPrecision(List<String> hyp, List<List<String>> refs, int n) {
        Map<String, Integer> hypNgrams = countNgrams(hyp, n);
        int clipped = 0, total = 0;
        for (var entry : hypNgrams.entrySet()) {
            int maxRef = refs.stream()
                .mapToInt(r -> countNgrams(r, n).getOrDefault(entry.getKey(), 0))
                .max().orElse(0);
            clipped += Math.min(entry.getValue(), maxRef);
            total   += entry.getValue();
        }
        return total > 0 ? (float) clipped / total : 0f;
    }

    private static Map<String, Integer> countNgrams(List<String> tokens, int n) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (int i = 0; i <= tokens.size() - n; i++) {
            String ngram = String.join(" ", tokens.subList(i, i + n));
            counts.merge(ngram, 1, Integer::sum);
        }
        return counts;
    }
}
