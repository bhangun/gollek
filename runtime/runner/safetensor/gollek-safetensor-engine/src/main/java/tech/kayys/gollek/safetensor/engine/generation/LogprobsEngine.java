/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * LogprobsEngine.java
 * ────────────────────
 * Computes per-token log probabilities for the OpenAI logprobs API.
 *
 * OpenAI logprobs response format
 * ════════════════════════════════
 * When the client sends {"logprobs": true, "top_logprobs": N}, each choice
 * in the response contains a "logprobs" object:
 *
 *   {
 *     "content": [
 *       {
 *         "token": "Hello",
 *         "logprob": -0.0123,
 *         "bytes": [72, 101, 108, 108, 111],
 *         "top_logprobs": [
 *           {"token": "Hello", "logprob": -0.0123, "bytes": [...]},
 *           {"token": "Hi",    "logprob": -2.47,   "bytes": [...]},
 *           ...
 *         ]
 *       },
 *       ...
 *     ]
 *   }
 *
 * logprob = log_softmax(logit[token_id])
 *
 * Integration
 * ═══════════
 * DirectInferenceEngine.runLoop() retains the softmax logits at each decode
 * step when logprobs are requested (GenerationConfig.returnLogprobs == true).
 * LogprobsEngine.compute() converts them to the OpenAI-format list.
 *
 * Usage
 * ═════
 *   List<TokenLogprob> logprobs = logprobsEngine.compute(
 *       generatedTokenIds, retainedLogits, tokenizer, topN);
 */
package tech.kayys.gollek.safetensor.engine.generation;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;

import org.jboss.logging.Logger;

import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Per-token log-probability computation for the OpenAI logprobs API.
 *
 * <p>
 * Inject and call {@link #compute} after generation to produce the
 * {@code logprobs.content} array.
 */
@ApplicationScoped
public class LogprobsEngine {

    private static final Logger log = Logger.getLogger(LogprobsEngine.class);

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Compute per-token logprobs from the retained softmax distributions.
     *
     * @param generatedIds the token IDs that were sampled (length = outputTokens)
     * @param logitsList   raw logits at each decode step (before softmax)
     *                     shape: [outputTokens][vocabSize]
     * @param tokenizer    for decoding token IDs to strings
     * @param topN         how many top-logprob alternatives to return per token
     * @return list of {@link TokenLogprob}, one per generated token
     */
    public List<TokenLogprob> compute(int[] generatedIds,
            List<float[]> logitsList,
            Tokenizer tokenizer,
            int topN) {
        if (generatedIds == null || logitsList == null || generatedIds.length == 0) {
            return List.of();
        }

        List<TokenLogprob> result = new ArrayList<>(generatedIds.length);

        for (int step = 0; step < generatedIds.length && step < logitsList.size(); step++) {
            int tokenId = generatedIds[step];
            float[] logits = logitsList.get(step);

            // Compute log-softmax
            float[] logprobs = logSoftmax(logits);

            // Sampled token
            String tokenStr = decode(tokenizer, tokenId);
            float lp = tokenId < logprobs.length ? logprobs[tokenId] : Float.NEGATIVE_INFINITY;

            // Top-N alternatives
            List<TopLogprob> top = computeTopN(logprobs, tokenizer, topN);

            result.add(new TokenLogprob(
                    tokenStr,
                    lp,
                    utf8Bytes(tokenStr),
                    top));
        }

        return result;
    }

    /**
     * Check if logprobs collection is needed based on request parameters.
     * Returns true when either {@code logprobs=true} or {@code top_logprobs > 0}.
     */
    public static boolean needed(boolean logprobs, int topLogprobs) {
        return logprobs || topLogprobs > 0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Math
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Numerically stable log-softmax: log(exp(x)/sum(exp(x))) = x -
     * log(sum(exp(x-max))).
     */
    private static float[] logSoftmax(float[] logits) {
        float max = logits[0];
        for (float v : logits)
            if (v > max)
                max = v;

        double sumExp = 0.0;
        for (float v : logits)
            sumExp += Math.exp(v - max);
        float logSumExp = (float) Math.log(sumExp);

        float[] lp = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            lp[i] = logits[i] - max - logSumExp;
        }
        return lp;
    }

    private List<TopLogprob> computeTopN(float[] logprobs, Tokenizer tokenizer, int n) {
        if (n <= 0)
            return List.of();

        // Partial sort: find top-N by log probability
        int vocabSize = logprobs.length;
        int k = Math.min(n, vocabSize);

        // Use a min-heap of size k
        PriorityQueue<int[]> heap = new PriorityQueue<>(k,
                Comparator.comparingDouble(a -> logprobs[a[0]]));

        for (int i = 0; i < vocabSize; i++) {
            if (heap.size() < k) {
                heap.offer(new int[] { i });
            } else if (logprobs[i] > logprobs[heap.peek()[0]]) {
                heap.poll();
                heap.offer(new int[] { i });
            }
        }

        // Sort descending
        List<int[]> topK = new ArrayList<>(heap);
        topK.sort((a, b) -> Float.compare(logprobs[b[0]], logprobs[a[0]]));

        List<TopLogprob> result = new ArrayList<>(topK.size());
        for (int[] entry : topK) {
            int id = entry[0];
            String tok = decode(tokenizer, id);
            result.add(new TopLogprob(tok, logprobs[id], utf8Bytes(tok)));
        }
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────

    private static String decode(Tokenizer tokenizer, int tokenId) {
        try {
            return tokenizer.decode(new long[] { tokenId }, DecodeOptions.defaultOptions());
        } catch (Exception e) {
            return "<" + tokenId + ">";
        }
    }

    private static List<Integer> utf8Bytes(String text) {
        if (text == null || text.isEmpty())
            return List.of();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        List<Integer> result = new ArrayList<>(bytes.length);
        for (byte b : bytes)
            result.add(b & 0xFF);
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // OpenAI response types
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Per-token log probability entry (OpenAI format).
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record TokenLogprob(
            @JsonProperty("token") String token,
            @JsonProperty("logprob") float logprob,
            @JsonProperty("bytes") List<Integer> bytes,
            @JsonProperty("top_logprobs") List<TopLogprob> topLogprobs) {
    }

    /**
     * Alternative token with its log probability.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record TopLogprob(
            @JsonProperty("token") String token,
            @JsonProperty("logprob") float logprob,
            @JsonProperty("bytes") List<Integer> bytes) {
    }

    /**
     * The {@code logprobs} object in a chat choice.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record LogprobsResult(
            @JsonProperty("content") List<TokenLogprob> content) {
        public static LogprobsResult of(List<TokenLogprob> content) {
            return new LogprobsResult(content);
        }
    }
}
