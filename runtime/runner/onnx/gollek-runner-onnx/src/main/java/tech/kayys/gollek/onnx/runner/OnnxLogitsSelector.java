package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class OnnxLogitsSelector {

    private final Ops ops;
    private final int vocabSize;
    private final boolean cacheTailPlan;
    private OnnxRuntimeBinding.LogitsTailPlan tailPlan;

    private OnnxLogitsSelector(Ops ops, int vocabSize, boolean cacheTailPlan) {
        this.ops = Objects.requireNonNull(ops, "ops");
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("vocabSize must be > 0");
        }
        this.vocabSize = vocabSize;
        this.cacheTailPlan = cacheTailPlan;
    }

    static OnnxLogitsSelector create(OnnxRuntimeBinding binding, int vocabSize, boolean cacheTailPlan) {
        return new OnnxLogitsSelector(new BindingOps(binding), vocabSize, cacheTailPlan);
    }

    static OnnxLogitsSelector createForTest(Ops ops, int vocabSize, boolean cacheTailPlan) {
        return new OnnxLogitsSelector(ops, vocabSize, cacheTailPlan);
    }

    int selectNextToken(MemorySegment logits, long sequenceLength) {
        Objects.requireNonNull(logits, "logits");
        if (sequenceLength <= 0) {
            throw new IllegalArgumentException("sequenceLength must be > 0");
        }
        if (!cacheTailPlan) {
            return ops.argmaxLast(logits, vocabSize);
        }
        if (tailPlan == null) {
            tailPlan = ops.createTailPlan(logits, vocabSize, sequenceLength);
        }
        return ops.argmaxLast(logits, tailPlan, sequenceLength);
    }

    interface Ops {
        OnnxRuntimeBinding.LogitsTailPlan createTailPlan(
                MemorySegment logits,
                int fallbackWidth,
                long sequenceLength);

        int argmaxLast(MemorySegment logits, int fallbackWidth);

        int argmaxLast(
                MemorySegment logits,
                OnnxRuntimeBinding.LogitsTailPlan tailPlan,
                long sequenceLength);
    }

    private record BindingOps(OnnxRuntimeBinding binding) implements Ops {
        private BindingOps {
            Objects.requireNonNull(binding, "binding");
        }

        @Override
        public OnnxRuntimeBinding.LogitsTailPlan createTailPlan(
                MemorySegment logits,
                int fallbackWidth,
                long sequenceLength) {
            return binding.createLogitsTailPlan(logits, fallbackWidth, sequenceLength);
        }

        @Override
        public int argmaxLast(MemorySegment logits, int fallbackWidth) {
            return binding.argmaxTensorDataFloatLast(logits, fallbackWidth);
        }

        @Override
        public int argmaxLast(
                MemorySegment logits,
                OnnxRuntimeBinding.LogitsTailPlan tailPlan,
                long sequenceLength) {
            return binding.argmaxTensorDataFloatLast(logits, tailPlan, sequenceLength);
        }
    }
}
