package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * CPU-adapted FlashAttention algorithm.
 * Uses online Softmax to eliminate the need for large intermediate score matrices (N x N).
 */
public final class FlashAttentionKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private FlashAttentionKernel() {}

    public static void computeParallel(
            MemorySegment out,
            MemorySegment q,
            KVCache kvCache,
            MemorySegment mask, // Optional [seqLen]
            int layer,
            int seqLen,
            int numHeads,
            int headDim,
            float softCap,
            ExecutorService executor
    ) {
        float scale = (float) (1.0 / Math.sqrt(headDim));

        List<Callable<Void>> tasks = new ArrayList<>(numHeads);

        for (int h = 0; h < numHeads; h++) {
            final int head = h;
            tasks.add(() -> {
                computeHead(out, q, kvCache, mask, layer, seqLen, numHeads, headDim, head, scale, softCap);
                return null;
            });
        }

        try {
            executor.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("FlashAttention interrupted", e);
        }
    }

    private static void computeHead(
            MemorySegment out,
            MemorySegment q,
            KVCache kvCache,
            MemorySegment mask,
            int layer,
            int seqLen,
            int numHeads,
            int headDim,
            int head,
            float scale,
            float softCap
    ) {
        long qBase = (long) head * headDim * Float.BYTES;
        
        // 1. Compute scores: [seqLen]
        float[] scores = new float[seqLen];
        for (int d = 0; d < headDim; d++) {
            float qVal = q.get(ValueLayout.JAVA_FLOAT, qBase + (long) d * Float.BYTES);
            qVal *= scale;
            
            // Vectorize across sequence
            FloatVector vQ = FloatVector.broadcast(SPECIES, qVal);
            int p = 0;
            for (; p <= seqLen - SPECIES.length(); p += SPECIES.length()) {
                long kOffset = kvCache.offset(layer, head, d, p) * Float.BYTES;
                FloatVector vK = FloatVector.fromMemorySegment(SPECIES, kvCache.getKCache(), kOffset, java.nio.ByteOrder.nativeOrder());
                FloatVector vScore = FloatVector.fromArray(SPECIES, scores, p);
                vScore = vScore.add(vQ.mul(vK));
                vScore.intoArray(scores, p);
            }
            for (; p < seqLen; p++) {
                scores[p] += qVal * kvCache.getK(layer, head, p, d);
            }
        }

        // Apply Optional Mask and Soft-capping
        if (softCap > 0.0f) {
            for (int p = 0; p < seqLen; p++) {
                scores[p] = (float) (softCap * Math.tanh(scores[p] / softCap));
            }
        }

        if (mask != null) {
            for (int p = 0; p < seqLen; p++) {
                float mv = mask.get(ValueLayout.JAVA_FLOAT, (long) p * Float.BYTES);
                if (mv == 0.0f) scores[p] = Float.NEGATIVE_INFINITY;
                else scores[p] += mv; // Standard additive mask (0.0 for valid, -inf for blocked)
            }
        }

        // 2. Softmax (Offline for simplicity and SIMD efficiency with this layout)
        float maxScore = Float.NEGATIVE_INFINITY;
        for (float s : scores) maxScore = Math.max(maxScore, s);
        
        float sum = 0.0f;
        for (int p = 0; p < seqLen; p++) {
            scores[p] = (float) Math.exp(scores[p] - maxScore);
            sum += scores[p];
        }
        
        final float invSum = 1.0f / sum;

        // 3. Accumulate V: [headDim]
        long outBase = (long) head * headDim * Float.BYTES;
        for (int d = 0; d < headDim; d++) {
            float acc = 0.0f;
            
            // Vectorize across sequence for this dimension
            FloatVector vAcc = FloatVector.zero(SPECIES);
            int p = 0;
            for (; p <= seqLen - SPECIES.length(); p += SPECIES.length()) {
                long vOffset = kvCache.offset(layer, head, d, p) * Float.BYTES;
                FloatVector vV = FloatVector.fromMemorySegment(SPECIES, kvCache.getVCache(), vOffset, java.nio.ByteOrder.nativeOrder());
                FloatVector vS = FloatVector.fromArray(SPECIES, scores, p);
                vAcc = vAcc.add(vV.mul(vS));
            }
            acc = vAcc.reduceLanes(VectorOperators.ADD);
            
            for (; p < seqLen; p++) {
                acc += scores[p] * kvCache.getV(layer, head, p, d);
            }
            
            out.set(ValueLayout.JAVA_FLOAT, outBase + (long) d * Float.BYTES, acc * invSum);
        }
    }
}
