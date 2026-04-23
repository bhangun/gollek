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
 * Applies a linear projection and accumulates the result into a residual tensor.
 * Formula: out = residual + bias + (weight * in)
 * Optimized to reduce task submission overhead.
 */
public final class LinearResidualKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private LinearResidualKernel() {}

    public static void computeParallel(
            MemorySegment in,         // [inDim]
            tech.kayys.gollek.gguf.loader.TensorData weight, // [outDim, inDim]
            MemorySegment bias,       // [outDim] or null
            MemorySegment residual,   // [outDim]
            MemorySegment out,        // [outDim] (can be the same as residual)
            int inDim,
            int outDim,
            ExecutorService executor
    ) {
        int numTasks = Math.min(Runtime.getRuntime().availableProcessors(), (outDim + 255) / 256);
        int step = (outDim + numTasks - 1) / numTasks;

        List<Callable<Void>> tasks = new ArrayList<>();
        for (int start = 0; start < outDim; start += step) {
            final int s = start;
            final int e = Math.min(start + step, outDim);
            tasks.add(() -> {
                try (java.lang.foreign.Arena taskArena = java.lang.foreign.Arena.ofConfined()) {
                    MemorySegment rowF32 = taskArena.allocate((long) inDim * Float.BYTES, 64);
                    for (int h = s; h < e; h++) {
                        float sum = 0;
                        if (weight.isQ8_0()) {
                            long bytesPerRow = (inDim / 32) * 34L;
                            tech.kayys.gollek.gguf.loader.GGUFDequantizer.dequantizeQ8_0(weight.segment(), h * bytesPerRow, rowF32, inDim);
                            sum = dot(in, rowF32, 0, inDim);
                        } else if (weight.isF16()) {
                            tech.kayys.gollek.gguf.loader.GGUFDequantizer.dequantizeF16(weight.segment(), (long) h * inDim * 2L, rowF32, inDim);
                            sum = dot(in, rowF32, 0, inDim);
                        } else {
                            sum = dot(in, weight.segment(), (long) h * inDim, inDim);
                        }
                        
                        float res = residual.get(ValueLayout.JAVA_FLOAT, (long) h * 4L);
                        float b = (bias != null) ? bias.get(ValueLayout.JAVA_FLOAT, (long) h * 4L) : 0.0f;
                        out.set(ValueLayout.JAVA_FLOAT, (long) h * 4L, res + b + sum);
                    }
                }
                return null;
            });
        }

        try {
            executor.invokeAll(tasks);
        } catch (InterruptedException err) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Linear projection interrupted", err);
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
