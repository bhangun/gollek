package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class OnnxPreparedRun {

    private final Ops ops;
    private final MemorySegment session;
    private final MemorySegment runOptions;
    private final MemorySegment inputNamePointers;
    private final MemorySegment inputValuePointers;
    private final MemorySegment outputNamePointers;
    private final MemorySegment outputValuePointers;
    private final OnnxRunOutputValues outputValues;

    private OnnxPreparedRun(
            Ops ops,
            MemorySegment session,
            MemorySegment runOptions,
            MemorySegment inputNamePointers,
            MemorySegment inputValuePointers,
            MemorySegment outputNamePointers,
            MemorySegment outputValuePointers,
            OnnxRunOutputValues outputValues) {
        this.ops = Objects.requireNonNull(ops, "ops");
        this.session = Objects.requireNonNull(session, "session");
        this.runOptions = Objects.requireNonNull(runOptions, "runOptions");
        this.inputNamePointers = Objects.requireNonNull(inputNamePointers, "inputNamePointers");
        this.inputValuePointers = Objects.requireNonNull(inputValuePointers, "inputValuePointers");
        this.outputNamePointers = Objects.requireNonNull(outputNamePointers, "outputNamePointers");
        this.outputValuePointers = Objects.requireNonNull(outputValuePointers, "outputValuePointers");
        this.outputValues = Objects.requireNonNull(outputValues, "outputValues");
    }

    static OnnxPreparedRun create(
            OnnxRuntimeBinding binding,
            MemorySegment session,
            MemorySegment inputNamePointers,
            MemorySegment inputValuePointers,
            MemorySegment outputNamePointers,
            MemorySegment outputValuePointers,
            OnnxRunOutputValues outputValues) {
        return new OnnxPreparedRun(
                new BindingOps(binding),
                session,
                MemorySegment.NULL,
                inputNamePointers,
                inputValuePointers,
                outputNamePointers,
                outputValuePointers,
                outputValues);
    }

    static OnnxPreparedRun createForTest(
            Ops ops,
            MemorySegment session,
            MemorySegment runOptions,
            MemorySegment inputNamePointers,
            MemorySegment inputValuePointers,
            MemorySegment outputNamePointers,
            MemorySegment outputValuePointers,
            OnnxRunOutputValues outputValues) {
        return new OnnxPreparedRun(
                ops,
                session,
                runOptions,
                inputNamePointers,
                inputValuePointers,
                outputNamePointers,
                outputValuePointers,
                outputValues);
    }

    OnnxRunOutputValues execute(
            OnnxRunInputValues inputValues,
            OnnxInferenceProfile profile,
            long inputPrepareStart,
            boolean prefill) {
        Objects.requireNonNull(inputValues, "inputValues");
        Objects.requireNonNull(profile, "profile");

        MemorySegment[] allInputValues = inputValues.completeValues();
        outputValues.prepareForRun();
        ops.writePointerArray(inputValuePointers, allInputValues, inputValues.count());
        profile.recordInputPrepare(inputPrepareStart);

        long runStart = profile.mark();
        ops.runWithPreparedPointers(
                session,
                runOptions,
                inputNamePointers,
                inputValuePointers,
                inputValues.count(),
                outputNamePointers,
                outputValuePointers,
                outputValues.values());
        outputValues.markRunCompleted();
        profile.recordOrtRun(runStart, prefill);
        return outputValues;
    }

    interface Ops {
        void writePointerArray(MemorySegment pointerArray, MemorySegment[] values, int count);

        void runWithPreparedPointers(
                MemorySegment session,
                MemorySegment runOptions,
                MemorySegment inputNamePointers,
                MemorySegment inputValuePointers,
                int inputCount,
                MemorySegment outputNamePointers,
                MemorySegment outputValuePointers,
                MemorySegment[] outputValues);
    }

    private record BindingOps(OnnxRuntimeBinding binding) implements Ops {
        private BindingOps {
            Objects.requireNonNull(binding, "binding");
        }

        @Override
        public void writePointerArray(MemorySegment pointerArray, MemorySegment[] values, int count) {
            binding.writePointerArrayUnchecked(pointerArray, values, count);
        }

        @Override
        public void runWithPreparedPointers(
                MemorySegment session,
                MemorySegment runOptions,
                MemorySegment inputNamePointers,
                MemorySegment inputValuePointers,
                int inputCount,
                MemorySegment outputNamePointers,
                MemorySegment outputValuePointers,
                MemorySegment[] outputValues) {
            binding.runWithPreparedPointers(
                    session,
                    runOptions,
                    inputNamePointers,
                    inputValuePointers,
                    inputCount,
                    outputNamePointers,
                    outputValuePointers,
                    outputValues);
        }
    }
}
