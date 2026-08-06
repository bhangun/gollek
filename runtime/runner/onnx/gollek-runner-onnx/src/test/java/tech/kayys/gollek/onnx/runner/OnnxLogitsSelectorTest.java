package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxLogitsSelectorTest {

    @Test
    void cachesTailPlanForKvDecode() {
        FakeOps ops = new FakeOps();
        OnnxLogitsSelector selector = OnnxLogitsSelector.createForTest(ops, 32_000, true);

        assertEquals(105, selector.selectNextToken(MemorySegment.NULL, 5));
        assertEquals(101, selector.selectNextToken(MemorySegment.NULL, 1));

        assertEquals(1, ops.createTailPlanCalls);
        assertEquals(0, ops.statelessArgmaxCalls);
        assertEquals(2, ops.cachedArgmaxCalls);
        assertEquals(List.of(5L), ops.planSequenceLengths);
        assertEquals(List.of(5L, 1L), ops.cachedArgmaxSequenceLengths);
        assertEquals(32_000, ops.lastFallbackWidth);
        assertSame(ops.plan, ops.lastCachedPlan);
    }

    @Test
    void usesStatelessArgmaxWithoutTailPlan() {
        FakeOps ops = new FakeOps();
        OnnxLogitsSelector selector = OnnxLogitsSelector.createForTest(ops, 4096, false);

        assertEquals(4096, selector.selectNextToken(MemorySegment.NULL, 7));

        assertEquals(0, ops.createTailPlanCalls);
        assertEquals(1, ops.statelessArgmaxCalls);
        assertEquals(0, ops.cachedArgmaxCalls);
        assertEquals(4096, ops.lastFallbackWidth);
    }

    @Test
    void validatesInputs() {
        FakeOps ops = new FakeOps();

        assertThrows(IllegalArgumentException.class, () -> OnnxLogitsSelector.createForTest(ops, 0, true));

        OnnxLogitsSelector selector = OnnxLogitsSelector.createForTest(ops, 1, true);
        assertThrows(NullPointerException.class, () -> selector.selectNextToken(null, 1));
        assertThrows(IllegalArgumentException.class, () -> selector.selectNextToken(MemorySegment.NULL, 0));
    }

    private static final class FakeOps implements OnnxLogitsSelector.Ops {
        private final OnnxRuntimeBinding.LogitsTailPlan plan = new OnnxRuntimeBinding.LogitsTailPlan(
                OnnxRuntimeBinding.ONNX_TENSOR_FLOAT,
                32_000,
                true);
        private final List<Long> planSequenceLengths = new ArrayList<>();
        private final List<Long> cachedArgmaxSequenceLengths = new ArrayList<>();
        private int createTailPlanCalls;
        private int statelessArgmaxCalls;
        private int cachedArgmaxCalls;
        private int lastFallbackWidth;
        private OnnxRuntimeBinding.LogitsTailPlan lastCachedPlan;

        @Override
        public OnnxRuntimeBinding.LogitsTailPlan createTailPlan(
                MemorySegment logits,
                int fallbackWidth,
                long sequenceLength) {
            createTailPlanCalls++;
            lastFallbackWidth = fallbackWidth;
            planSequenceLengths.add(sequenceLength);
            return plan;
        }

        @Override
        public int argmaxLast(MemorySegment logits, int fallbackWidth) {
            statelessArgmaxCalls++;
            lastFallbackWidth = fallbackWidth;
            return fallbackWidth;
        }

        @Override
        public int argmaxLast(
                MemorySegment logits,
                OnnxRuntimeBinding.LogitsTailPlan tailPlan,
                long sequenceLength) {
            cachedArgmaxCalls++;
            lastCachedPlan = tailPlan;
            cachedArgmaxSequenceLengths.add(sequenceLength);
            return 100 + (int) sequenceLength;
        }
    }
}
