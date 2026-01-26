package tech.kayys.golek.engine;

import java.util.List;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;

import java.util.Map;

public class Sampler {
    private final SamplingParams params;
    private final Random rng;
    private final List<Integer> lastTokens;
    
    public Sampler(SamplingParams params) {
        this.params = params;
        this.rng = params.seed() >= 0 ? new Random(params.seed()) : new Random();
        this.lastTokens = new ArrayList<>(params.repeatLastN());
    }
    
    public int sample(Arena arena, MemorySegment logits, int vocabSize, Set<Integer> stopTokens) {
        // Copy logits to array
        float[] logitsArray = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            logitsArray[i] = logits.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        
        // Apply repeat penalty
        if (params.repeatPenalty() != 1.0f && !lastTokens.isEmpty()) {
            for (int token : lastTokens) {
                if (logitsArray[token] > 0) {
                    logitsArray[token] /= params.repeatPenalty();
                } else {
                    logitsArray[token] *= params.repeatPenalty();
                }
            }
        }
        
        // Apply frequency and presence penalties
        if (params.frequencyPenalty() != 0.0f || params.presencePenalty() != 0.0f) {
            Map<Integer, Integer> tokenCounts = new HashMap<>();
            for (int token : lastTokens) {
                tokenCounts.merge(token, 1, Integer::sum);
            }
            
            for (Map.Entry<Integer, Integer> entry : tokenCounts.entrySet()) {
                int token = entry.getKey();
                int count = entry.getValue();
                float penalty = params.frequencyPenalty() * count + params.presencePenalty();
                logitsArray[token] -= penalty;
            }
        }
        
        // Top-K filtering
        if (params.topK() > 0 && params.topK() < vocabSize) {
            float[] sortedLogits = logitsArray.clone();
            Arrays.sort(sortedLogits);
            float kthLargest = sortedLogits[vocabSize - params.topK()];
            for (int i = 0; i < vocabSize; i++) {
                if (logitsArray[i] < kthLargest) {
                    logitsArray[i] = Float.NEGATIVE_INFINITY;
                }
            }
        }
        
        // Temperature scaling
        if (params.temperature() > 0 && params.temperature() != 1.0f) {
            for (int i = 0; i < vocabSize; i++) {
                logitsArray[i] /= params.temperature();
            }
        }
        
        // Softmax
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float logit : logitsArray) {
            if (logit > maxLogit && !Float.isInfinite(logit)) {
                maxLogit = logit;
            }
        }
        
        float sumExp = 0.0f;
        float[] probs = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            if (Float.isInfinite(logitsArray[i])) {
                probs[i] = 0.0f;
            } else {
                probs[i] = (float) Math.exp(logitsArray[i] - maxLogit);
                sumExp += probs[i];
            }
        }
        
        for (int i = 0; i < vocabSize; i++) {
            probs[i] /= sumExp;
        }
        
        // Min-P filtering
        if (params.minP() > 0.0f) {
            float maxProb = 0.0f;
            for (float p : probs) {
                if (p > maxProb) maxProb = p;
            }
            float threshold = params.minP() * maxProb;
            for (int i = 0; i < vocabSize; i++) {
                if (probs[i] < threshold) {
                    probs[i] = 0.0f;
                }
            }
            
            // Renormalize
            sumExp = 0.0f;
            for (float p : probs) sumExp += p;
            if (sumExp > 0) {
                for (int i = 0; i < vocabSize; i++) {
                    probs[i] /= sumExp;
                }
            }
        }
        
        // Top-P (nucleus) sampling
        if (params.topP() < 1.0f) {
            List<TokenProb> sorted = new ArrayList<>();
            for (int i = 0; i < vocabSize; i++) {
                if (probs[i] > 0) {
                    sorted.add(new TokenProb(i, probs[i]));
                }
            }
            sorted.sort((a, b) -> Float.compare(b.prob, a.prob));
            
            float cumProb = 0.0f;
            int cutoff = 0;
            for (int i = 0; i < sorted.size(); i++) {
                cumProb += sorted.get(i).prob;
                if (cumProb >= params.topP()) {
                    cutoff = i + 1;
                    break;
                }
            }
            
            Arrays.fill(probs, 0.0f);
            sumExp = 0.0f;
            for (int i = 0; i < cutoff; i++) {
                TokenProb tp = sorted.get(i);
                probs[tp.token] = tp.prob;
                sumExp += tp.prob;
            }
            
            // Renormalize
            if (sumExp > 0) {
                for (int i = 0; i < vocabSize; i++) {
                    probs[i] /= sumExp;
                }
            }
        }
        
        // Sample from distribution
        float random = rng.nextFloat();
        float cumulative = 0.0f;
        for (int i = 0; i < vocabSize; i++) {
            cumulative += probs[i];
            if (random < cumulative) {
                accept(i);
                return i;
            }
        }
        
        // Fallback to last token with non-zero probability
        for (int i = vocabSize - 1; i >= 0; i--) {
            if (probs[i] > 0) {
                accept(i);
                return i;
            }
        }
        
        accept(0);
        return 0;
    }
    
    public void accept(int token) {
        lastTokens.add(token);
        if (lastTokens.size() > params.repeatLastN()) {
            lastTokens.remove(0);
        }
    }
    
    public void reset() {
        lastTokens.clear();
    }
    
    private record TokenProb(int token, float prob) {}
}
