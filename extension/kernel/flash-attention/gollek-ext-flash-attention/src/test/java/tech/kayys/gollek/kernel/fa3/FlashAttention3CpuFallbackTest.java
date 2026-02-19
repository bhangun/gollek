package tech.kayys.gollek.kernel.fa3;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

class FlashAttention3CpuFallbackTest {

    @Test
    void testStandardAttentionFallback() {
        int batchSize = 1;
        int seqLen = 2;
        int numHeads = 1;
        int numHeadsK = 1; // standard MHA
        int headDim = 2;
        float softmaxScale = 1.0f;
        boolean isCausal = false;

        try (Arena arena = Arena.ofConfined()) {
            long floats = (long) batchSize * seqLen * numHeads * headDim;
            MemorySegment q = arena.allocate(floats * 4L);
            MemorySegment k = arena.allocate(floats * 4L);
            MemorySegment v = arena.allocate(floats * 4L);
            MemorySegment o = arena.allocate(floats * 4L);

            ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT;

            // Simple data:
            // Q = [[1, 0], [0, 1]]
            // K = [[1, 0], [0, 1]]
            // V = [[1, 1], [2, 2]]
            // Softmax scores matrix should be:
            // S_11 = exp(1) S_12 = exp(0)
            // S_21 = exp(0) S_22 = exp(1)

            q.set(FLOAT, 0 * 4L, 1.0f); q.set(FLOAT, 1 * 4L, 0.0f);
            q.set(FLOAT, 2 * 4L, 0.0f); q.set(FLOAT, 3 * 4L, 1.0f);

            k.set(FLOAT, 0 * 4L, 1.0f); k.set(FLOAT, 1 * 4L, 0.0f);
            k.set(FLOAT, 2 * 4L, 0.0f); k.set(FLOAT, 3 * 4L, 1.0f);

            v.set(FLOAT, 0 * 4L, 1.0f); v.set(FLOAT, 1 * 4L, 1.0f);
            v.set(FLOAT, 2 * 4L, 2.0f); v.set(FLOAT, 3 * 4L, 2.0f);

            FlashAttention3CpuFallback.execute(
                    o, q, k, v, 
                    batchSize, seqLen, numHeads, numHeadsK, headDim, 
                    softmaxScale, isCausal, false
            );

            // Calculation check:
            // row 0: Q=[1, 0] -> dot with K0=[1, 0] -> 1. dot with K1=[0, 1] -> 0.
            // Softmax(1, 0) = [e^1/(e^1+e^0), e^0/(e^1+e^0)] = [2.718/3.718, 1/3.718] = [0.731, 0.269]
            // O0 = 0.731 * [1, 1] + 0.269 * [2, 2] = [0.731 + 0.538, 0.731 + 0.538] = [1.269, 1.269]
            
            float expectedVal0 = (float) (Math.exp(1) * 1 + Math.exp(0) * 2) / (float) (Math.exp(1) + Math.exp(0));
            float expectedVal1 = (float) (Math.exp(0) * 1 + Math.exp(1) * 2) / (float) (Math.exp(1) + Math.exp(0));

            assertThat(o.get(FLOAT, 0 * 4L)).isCloseTo(expectedVal0, offset(0.001f));
            assertThat(o.get(FLOAT, 1 * 4L)).isCloseTo(expectedVal0, offset(0.001f));
            assertThat(o.get(FLOAT, 2 * 4L)).isCloseTo(expectedVal1, offset(0.001f));
            assertThat(o.get(FLOAT, 3 * 4L)).isCloseTo(expectedVal1, offset(0.001f));
        }
    }

    @Test
    void testStandardAttentionFallbackCausal() {
        int batchSize = 1;
        int seqLen = 2;
        int numHeads = 1;
        int numHeadsK = 1;
        int headDim = 2;
        float softmaxScale = 1.0f;
        boolean isCausal = true;

        try (Arena arena = Arena.ofConfined()) {
            long floats = (long) batchSize * seqLen * numHeads * headDim;
            MemorySegment q = arena.allocate(floats * 4L);
            MemorySegment k = arena.allocate(floats * 4L);
            MemorySegment v = arena.allocate(floats * 4L);
            MemorySegment o = arena.allocate(floats * 4L);

            ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT;

            q.set(FLOAT, 0 * 4L, 1.0f); q.set(FLOAT, 1 * 4L, 0.0f);
            q.set(FLOAT, 2 * 4L, 0.0f); q.set(FLOAT, 3 * 4L, 1.0f);

            k.set(FLOAT, 0 * 4L, 1.0f); k.set(FLOAT, 1 * 4L, 0.0f);
            k.set(FLOAT, 2 * 4L, 0.0f); k.set(FLOAT, 3 * 4L, 1.0f);

            v.set(FLOAT, 0 * 4L, 1.0f); v.set(FLOAT, 1 * 4L, 1.0f);
            v.set(FLOAT, 2 * 4L, 2.0f); v.set(FLOAT, 3 * 4L, 2.0f);

            FlashAttention3CpuFallback.execute(
                    o, q, k, v, 
                    batchSize, seqLen, numHeads, numHeadsK, headDim, 
                    softmaxScale, isCausal, false
            );

            // Causal:
            // q=0 can only see k=0
            // score = [1, -inf] -> softmax = [1, 0]
            // O0 = 1 * [1, 1] + 0 = [1.0, 1.0]
            
            // q=1 can see k=0, 1
            // same as test above
            float expectedVal1 = (float) (Math.exp(0) * 1 + Math.exp(1) * 2) / (float) (Math.exp(1) + Math.exp(0));

            assertThat(o.get(FLOAT, 0 * 4L)).isCloseTo(1.0f, offset(0.001f));
            assertThat(o.get(FLOAT, 1 * 4L)).isCloseTo(1.0f, offset(0.001f));
            assertThat(o.get(FLOAT, 2 * 4L)).isCloseTo(expectedVal1, offset(0.001f));
            assertThat(o.get(FLOAT, 3 * 4L)).isCloseTo(expectedVal1, offset(0.001f));
        }
    }
}
