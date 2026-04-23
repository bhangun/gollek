package tech.kayys.gollek.inference.nativeimpl;

import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized Logit Projection Kernel.
 * Computes: [1, hidden] x [hidden, vocab_size] -> [1, vocab_size]
 * <p>
 * This is effectively a large batch of dot products.
 */
public final class LogitProjectionKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void execute(
        MemorySegment x,          // [hidden]
        tech.kayys.gollek.gguf.loader.TensorData outputWeight, // [vocab_size, hidden]
        MemorySegment logits,       // [vocab_size]
        int hidden,
        int vocabSize,
        float softCap
    ) {
        int tasksCount = Runtime.getRuntime().availableProcessors();
        int step = (vocabSize + tasksCount - 1) / tasksCount;

        java.util.stream.IntStream.range(0, tasksCount).parallel().forEach(t -> {
            int start = t * step;
            int end = Math.min(start + step, vocabSize);
            if (start >= end) return;

            try (java.lang.foreign.Arena taskArena = java.lang.foreign.Arena.ofConfined()) {
                MemorySegment rowF32 = taskArena.allocate((long) hidden * Float.BYTES, 64);
                
                for (int v = start; v < end; v++) {
                    if (outputWeight.isQ8_0()) {
                        long bytesPerRow = (hidden / 32) * 34L;
                        tech.kayys.gollek.gguf.loader.GGUFDequantizer.dequantizeQ8_0(outputWeight.segment(), (long) v * bytesPerRow, rowF32, hidden);
                    } else if (outputWeight.isF16()) {
                        tech.kayys.gollek.gguf.loader.GGUFDequantizer.dequantizeF16(outputWeight.segment(), (long) v * hidden * 2L, rowF32, hidden);
                    } else {
                        MemorySegment.copy(outputWeight.segment(), (long) v * hidden * Float.BYTES, rowF32, 0, (long) hidden * Float.BYTES);
                    }
                    
                    float sum = 0.0f;
                    int i = 0;
                    int upperBound = SPECIES.loopBound(hidden);
                    FloatVector acc = FloatVector.zero(SPECIES);
                    
                    for (; i < upperBound; i += SPECIES.length()) {
                        FloatVector vx = FloatVector.fromMemorySegment(SPECIES, x, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                        FloatVector vw = FloatVector.fromMemorySegment(SPECIES, rowF32, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                        acc = vx.fma(vw, acc);
                    }
                    
                    sum = acc.reduceLanes(VectorOperators.ADD);
                    for (; i < hidden; i++) {
                        sum += x.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES) * 
                               rowF32.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
                    }
                    
                    if (softCap > 0.0f) {
                        sum = (float) (softCap * Math.tanh(sum / softCap));
                    }
                    
                    logits.set(ValueLayout.JAVA_FLOAT, (long) v * Float.BYTES, sum);
                }
            }
        });
    }
}
