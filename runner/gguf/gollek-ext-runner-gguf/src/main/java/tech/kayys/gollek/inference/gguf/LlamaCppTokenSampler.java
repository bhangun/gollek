package tech.kayys.gollek.inference.gguf;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

/**
 * Handles token sampling strategies including temperature scaling, top-k, top-p,
 * min-p filtering, and penalty application (repeat, frequency, presence).
 */
public class LlamaCppTokenSampler {

    private final LlamaCppBinding binding;
    private final int vocabSize;
    private final ThreadLocal<TokenProb[]> tokenBufferLocal;

    public LlamaCppTokenSampler(LlamaCppBinding binding, int vocabSize) {
        this.binding = binding;
        this.vocabSize = vocabSize;
        this.tokenBufferLocal = ThreadLocal.withInitial(() -> new TokenProb[0]);
    }

    /**
     * Sample the next token from the model's logits using configured sampling strategy.
     */
    public int sampleNextToken(
            MemorySegment context,
            int batchIndex,
            SamplingConfig config,
            Random random) {

        MemorySegment logits = getLogits(context, batchIndex);
        if (logits == null || logits.equals(MemorySegment.NULL)) {
            throw new RuntimeException("No logits available for sampling");
        }

        int effectiveVocab = vocabSize > 0 ? vocabSize : 32768;
        logits = logits.reinterpret((long) effectiveVocab * Float.BYTES);

        if (config.temperature <= 0.0f) {
            return argMaxToken(logits, effectiveVocab);
        }

        TokenProb[] tokenBuffer = getTokenBuffer(effectiveVocab);
        int candidateCount = applyPenaltiesAndTemperature(
                tokenBuffer,
                logits,
                effectiveVocab,
                config);

        if (config.topK == 1) {
            return argMaxToken(tokenBuffer, candidateCount);
        }

        if ((config.topK <= 0) && (config.topP <= 0.0f || config.topP >= 1.0f) && config.minP <= 0.0f) {
            return sampleFromUnsorted(tokenBuffer, candidateCount, random);
        }

        return applyFilteringAndSample(tokenBuffer, candidateCount, config, random);
    }

    private MemorySegment getLogits(MemorySegment context, int batchIndex) {
        try {
            return binding.getLogitsIth(context, batchIndex);
        } catch (RuntimeException e) {
            return binding.getLogits(context);
        }
    }

    private int applyPenaltiesAndTemperature(
            TokenProb[] buffer,
            MemorySegment logits,
            int effectiveVocab,
            SamplingConfig config) {

        float invTemp = 1.0f / Math.max(config.temperature, 1.0e-6f);

        for (int i = 0; i < effectiveVocab; i++) {
            float value = logits.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            int count = config.recentTokenCounts == null ? 0 : config.recentTokenCounts[i];

            if (count > 0 && config.repeatPenalty > 1.0f) {
                value = value < 0.0f ? value * config.repeatPenalty : value / config.repeatPenalty;
            }
            if (count > 0 && config.presencePenalty != 0.0f) {
                value -= config.presencePenalty;
            }
            if (count > 0 && config.frequencyPenalty != 0.0f) {
                value -= config.frequencyPenalty * count;
            }

            buffer[i].logit = value * invTemp;
            buffer[i].tokenId = i;
        }

        return effectiveVocab;
    }

    private int applyFilteringAndSample(
            TokenProb[] buffer,
            int size,
            SamplingConfig config,
            Random random) {

        int k = config.topK > 0 ? Math.min(config.topK, size) : size;
        partialSelectTopK(buffer, size, k);
        size = k;

        float maxLogit = buffer[0].logit;
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            TokenProb c = buffer[i];
            c.prob = Math.exp(c.logit - maxLogit);
            sum += c.prob;
        }

        if (sum <= 0.0) {
            return buffer[0].tokenId;
        }

        for (int i = 0; i < size; i++) {
            buffer[i].prob /= sum;
        }

        if (config.topP > 0.0f && config.topP < 1.0f) {
            size = applyNucleusSampling(buffer, size, config.topP);
        }

        if (config.minP > 0.0f) {
            size = applyMinPFiltering(buffer, size, config.minP);
        }

        if (size <= 1) {
            return buffer[0].tokenId;
        }

        return sampleFromSorted(buffer, size, random);
    }

    private int applyNucleusSampling(TokenProb[] buffer, int size, float topP) {
        double cumulative = 0.0;
        int nucleusCount = 0;
        for (int i = 0; i < size; i++) {
            TokenProb c = buffer[i];
            cumulative += c.prob;
            nucleusCount++;
            if (cumulative >= topP) {
                break;
            }
        }
        if (nucleusCount < size) {
            size = nucleusCount;
            normalizeProbabilities(buffer, size);
        }
        return size;
    }

    private int applyMinPFiltering(TokenProb[] buffer, int size, float minP) {
        double best = buffer[0].prob;
        double threshold = best * minP;
        int filteredCount = 0;
        for (int i = 0; i < size; i++) {
            TokenProb c = buffer[i];
            if (c.prob >= threshold) {
                buffer[filteredCount++] = c;
            }
        }
        if (filteredCount == 0) {
            return buffer[0].tokenId;
        }
        if (filteredCount < size) {
            size = filteredCount;
            normalizeProbabilities(buffer, size);
        }
        return size;
    }

    private int sampleFromSorted(TokenProb[] candidates, int size, Random random) {
        double r = random.nextDouble();
        double acc = 0.0;
        for (int i = 0; i < size; i++) {
            TokenProb c = candidates[i];
            acc += c.prob;
            if (r <= acc) {
                return c.tokenId;
            }
        }
        return candidates[size - 1].tokenId;
    }

    private int sampleFromUnsorted(TokenProb[] candidates, int size, Random random) {
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            float value = candidates[i].logit;
            if (value > maxLogit) {
                maxLogit = value;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            double p = Math.exp(candidates[i].logit - maxLogit);
            candidates[i].prob = p;
            sum += p;
        }
        if (sum <= 0.0) {
            return candidates[0].tokenId;
        }

        double r = random.nextDouble() * sum;
        double acc = 0.0;
        for (int i = 0; i < size; i++) {
            acc += candidates[i].prob;
            if (r <= acc) {
                return candidates[i].tokenId;
            }
        }
        return candidates[size - 1].tokenId;
    }

    private int argMaxToken(MemorySegment logits, int effectiveVocab) {
        int bestId = 0;
        float best = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < effectiveVocab; i++) {
            float value = logits.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (value > best) {
                best = value;
                bestId = i;
            }
        }
        return bestId;
    }

    private int argMaxToken(TokenProb[] candidates, int size) {
        int bestId = candidates[0].tokenId;
        float best = candidates[0].logit;
        for (int i = 1; i < size; i++) {
            float value = candidates[i].logit;
            if (value > best) {
                best = value;
                bestId = candidates[i].tokenId;
            }
        }
        return bestId;
    }

    private void partialSelectTopK(TokenProb[] buffer, int length, int k) {
        if (k >= length) {
            java.util.Arrays.sort(buffer, 0, length, (a, b) -> Float.compare(b.logit, a.logit));
            return;
        }
        for (int i = 0; i < k; i++) {
            int best = i;
            float bestLogit = buffer[i].logit;
            for (int j = i + 1; j < length; j++) {
                float value = buffer[j].logit;
                if (value > bestLogit) {
                    bestLogit = value;
                    best = j;
                }
            }
            if (best != i) {
                TokenProb tmp = buffer[i];
                buffer[i] = buffer[best];
                buffer[best] = tmp;
            }
        }
        java.util.Arrays.sort(buffer, 0, k, (a, b) -> Float.compare(b.logit, a.logit));
    }

    private void normalizeProbabilities(TokenProb[] candidates, int size) {
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += candidates[i].prob;
        }
        if (sum <= 0.0) {
            return;
        }
        for (int i = 0; i < size; i++) {
            candidates[i].prob /= sum;
        }
    }

    private TokenProb[] getTokenBuffer(int size) {
        TokenProb[] buffer = tokenBufferLocal.get();
        if (buffer.length >= size) {
            return buffer;
        }
        TokenProb[] expanded = new TokenProb[size];
        int i = 0;
        for (; i < buffer.length; i++) {
            expanded[i] = buffer[i];
        }
        for (; i < size; i++) {
            expanded[i] = new TokenProb(i, 0.0f);
        }
        tokenBufferLocal.set(expanded);
        return expanded;
    }

    /**
     * Configuration for token sampling.
     */
    public static class SamplingConfig {
        public final float temperature;
        public final int topK;
        public final float topP;
        public final float minP;
        public final float repeatPenalty;
        public final float frequencyPenalty;
        public final float presencePenalty;
        public final int[] recentTokenCounts;

        public SamplingConfig(float temperature, int topK, float topP, float minP,
                             float repeatPenalty, float frequencyPenalty, float presencePenalty,
                             int[] recentTokenCounts) {
            this.temperature = temperature;
            this.topK = topK;
            this.topP = topP;
            this.minP = minP;
            this.repeatPenalty = repeatPenalty;
            this.frequencyPenalty = frequencyPenalty;
            this.presencePenalty = presencePenalty;
            this.recentTokenCounts = recentTokenCounts;
        }
    }

    private static final class TokenProb {
        private int tokenId;
        private float logit;
        private double prob;

        private TokenProb(int tokenId, float logit) {
            this.tokenId = tokenId;
            this.logit = logit;
        }
    }
}
