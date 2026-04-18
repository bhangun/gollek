package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import tech.kayys.gollek.gguf.loader.TransformerLayerWeights;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Fused FFN Kernel implementing SwiGLU with optional biases.
 * Optimized to avoid asSlice() overhead in hot loops.
 */
public final class FusedFFNKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private FusedFFNKernel() {}

    public static void computeParallel(
            MemorySegment x,          // [hidden]
            TransformerLayerWeights w,
            MemorySegment ffnState,   // [ffnDim]
            MemorySegment residual,   // [hidden]
            MemorySegment out,        // [hidden]
            int hidden,
            int ffnDim,
            ExecutorService executor
    ) {
        // Step 1: Up and Gate Projection (SwiGLU) -> stores in ffnState
        List<Callable<Void>> tasks1 = new ArrayList<>();
        int tasksCount1 = Runtime.getRuntime().availableProcessors() * 2;
        int step1 = (ffnDim + tasksCount1 - 1) / tasksCount1;

        for (int start = 0; start < ffnDim; start += step1) {
            final int s = start;
            final int e = Math.min(start + step1, ffnDim);
            tasks1.add(() -> {
                for (int f = s; f < e; f++) {
                    float sumG = dot(x, w.wG, (long) f * hidden, hidden);
                    if (w.bg != null) sumG += w.bg.get(ValueLayout.JAVA_FLOAT, (long) f * 4L);
                    
                    float sumU = dot(x, w.wU, (long) f * hidden, hidden);
                    if (w.bu != null) sumU += w.bu.get(ValueLayout.JAVA_FLOAT, (long) f * 4L);

                    float silu = sumG / (1.0f + (float) Math.exp(-sumG));
                    float val = silu * sumU;

                    ffnState.set(ValueLayout.JAVA_FLOAT, (long) f * 4L, val);
                }
                return null;
            });
        }

        try {
            executor.invokeAll(tasks1);
        } catch (InterruptedException err) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("FFN Up/Gate interrupted", err);
        }

        // Step 2: Down Projection -> stores in out
        List<Callable<Void>> tasks2 = new ArrayList<>();
        int step2 = (hidden + tasksCount1 - 1) / tasksCount1;

        for (int start = 0; start < hidden; start += step2) {
            final int s = start;
            final int e = Math.min(start + step2, hidden);
            tasks2.add(() -> {
                for (int h = s; h < e; h++) {
                    float sumD = dot(ffnState, w.wD, (long) h * ffnDim, ffnDim);
                    if (w.bd != null) sumD += w.bd.get(ValueLayout.JAVA_FLOAT, (long) h * 4L);
                    
                    float res = residual.get(ValueLayout.JAVA_FLOAT, (long) h * 4L);
                    out.set(ValueLayout.JAVA_FLOAT, (long) h * 4L, res + sumD);
                }
                return null;
            });
        }

        try {
            executor.invokeAll(tasks2);
        } catch (InterruptedException err) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("FFN Down interrupted", err);
        }
    }

    private static float dot(MemorySegment vec, MemorySegment mat, long matOffset, int colSize) {
        int i = 0;
        FloatVector vsum = FloatVector.zero(SPECIES);
        long matByteBase = matOffset * 4L;

        for (; i <= colSize - SPECIES.length(); i += SPECIES.length()) {
            var v1 = FloatVector.fromMemorySegment(SPECIES, vec, (long) i * 4L, java.nio.ByteOrder.nativeOrder());
            var v2 = FloatVector.fromMemorySegment(SPECIES, mat, matByteBase + (long) i * 4L, java.nio.ByteOrder.nativeOrder());
            vsum = v1.fma(v2, vsum);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);

        for (; i < colSize; i++) {
            float val1 = vec.get(ValueLayout.JAVA_FLOAT, (long) i * 4L);
            float val2 = mat.get(ValueLayout.JAVA_FLOAT, matByteBase + (long) i * 4L);
            sum += val1 * val2;
        }

        return sum;
    }
}
