package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

class OnnxTextRunWorkspaceTest {

    @Test
    void drivesPreparedRunLogitSelectionAndOwnedInputRelease() {
        try (Arena arena = Arena.ofConfined()) {
            List<MemorySegment> released = new ArrayList<>();
            FakeInputOps inputOps = new FakeInputOps(arena);
            MemorySegment logits = arena.allocate(Long.BYTES);
            OnnxRunOutputValues outputValues = OnnxRunOutputValues.allocate(1);
            OnnxTextRunWorkspace workspace = createWorkspace(
                    arena,
                    false,
                    2,
                    0,
                    inputOps,
                    new FakeRunOps(logits),
                    outputValues,
                    released);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 10, 20 });
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(false, false, 2, 2, 0);
            OnnxInferenceProfile profile = OnnxInferenceProfile.start(profiledRequest());

            workspace.beginRequest();
            OnnxRunInputValues inputValues = workspace.assembleInputs(tokens, 2, step);
            workspace.execute(inputValues, profile, profile.mark(), false);
            int next = workspace.selectNextToken(step.sequenceLength());
            workspace.releaseOwnedInputs(inputValues);
            workspace.capturePresentAndReleaseLogits();
            workspace.close();
            workspace.close();

            assertFalse(workspace.hasKvInputs());
            assertEquals(102, next);
            assertEquals(2, inputOps.createdTensors.size());
            assertEquals(List.of(
                    logits,
                    inputOps.createdTensors.get(0),
                    inputOps.createdTensors.get(1)), released);
        }
    }

    @Test
    void closeReleasesLazyEmptyPastKvOnce() {
        try (Arena arena = Arena.ofConfined()) {
            List<MemorySegment> released = new ArrayList<>();
            FakeInputOps inputOps = new FakeInputOps(arena);
            MemorySegment emptyPastKv = arena.allocate(Long.BYTES);
            OnnxRepeatedInputValue emptyPastKvInput = OnnxRepeatedInputValue.create(
                    2,
                    () -> emptyPastKv,
                    released::add);
            OnnxTextInputAssembler inputAssembler = createInputAssembler(
                    arena,
                    inputOps,
                    4,
                    false,
                    true,
                    2,
                    emptyPastKvInput);
            OnnxRunOutputValues outputValues = OnnxRunOutputValues.allocate(1);
            OnnxTextRunWorkspace workspace = OnnxTextRunWorkspace.createForTest(
                    8,
                    true,
                    OnnxPastKvState.allocate(2),
                    inputAssembler,
                    createPreparedRun(arena, new FakeRunOps(arena.allocate(Long.BYTES)), outputValues),
                    outputValues,
                    OnnxRunOutputLifecycle.create(outputValues, OnnxPastKvState.allocate(0), false, released::add),
                    OnnxLogitsSelector.createForTest(new FakeLogitsOps(), 4, false),
                    emptyPastKvInput,
                    released::add);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 1, 2 });
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(true, false, 2, 2, 0);

            workspace.beginRequest();
            workspace.assembleInputs(tokens, 2, step);
            workspace.close();
            workspace.close();

            assertEquals(List.of(
                    inputOps.createdTensors.get(0),
                    inputOps.createdTensors.get(1),
                    emptyPastKv), released);
        }
    }

    @Test
    void finishRequestKeepsLazyEmptyPastKvForWorkspaceReuse() {
        try (Arena arena = Arena.ofConfined()) {
            List<MemorySegment> released = new ArrayList<>();
            AtomicInteger created = new AtomicInteger();
            FakeInputOps inputOps = new FakeInputOps(arena);
            MemorySegment emptyPastKv = arena.allocate(Long.BYTES);
            OnnxRepeatedInputValue emptyPastKvInput = OnnxRepeatedInputValue.create(
                    2,
                    () -> {
                        created.incrementAndGet();
                        return emptyPastKv;
                    },
                    released::add);
            OnnxTextInputAssembler inputAssembler = createInputAssembler(
                    arena,
                    inputOps,
                    4,
                    false,
                    true,
                    2,
                    emptyPastKvInput);
            OnnxRunOutputValues outputValues = OnnxRunOutputValues.allocate(1);
            OnnxTextRunWorkspace workspace = OnnxTextRunWorkspace.createForTest(
                    8,
                    true,
                    OnnxPastKvState.allocate(2),
                    inputAssembler,
                    createPreparedRun(arena, new FakeRunOps(arena.allocate(Long.BYTES)), outputValues),
                    outputValues,
                    OnnxRunOutputLifecycle.create(outputValues, OnnxPastKvState.allocate(0), false, released::add),
                    OnnxLogitsSelector.createForTest(new FakeLogitsOps(), 4, false),
                    emptyPastKvInput,
                    released::add);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 1, 2 });
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(true, false, 2, 2, 0);

            workspace.beginRequest();
            workspace.assembleInputs(tokens, 2, step);
            workspace.finishRequest();
            workspace.beginRequest();
            workspace.assembleInputs(tokens, 2, step);
            workspace.finishRequest();

            assertEquals(1, created.get());
            assertEquals(List.of(), released);

            workspace.close();

            assertEquals(List.of(
                    inputOps.createdTensors.get(0),
                    inputOps.createdTensors.get(1),
                    emptyPastKv), released);
        }
    }

    @Test
    void reusesRequestTokenBuffersAcrossRequests() {
        try (Arena arena = Arena.ofConfined()) {
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunOutputValues outputValues = OnnxRunOutputValues.allocate(1);
            OnnxTextRunWorkspace workspace = createWorkspace(
                    arena,
                    false,
                    2,
                    0,
                    new FakeInputOps(arena),
                    new FakeRunOps(arena.allocate(Long.BYTES)),
                    outputValues,
                    released);

            workspace.beginRequest();
            OnnxTokenHistory firstHistory = workspace.resetTokenHistory(new int[] { 1, 2, 3 }, 2);
            OnnxGeneratedTokens firstGenerated = workspace.resetGeneratedTokens(2);
            firstGenerated.append(10);
            workspace.finishRequest();

            workspace.beginRequest();
            OnnxTokenHistory secondHistory = workspace.resetTokenHistory(new int[] { 7 }, 1);
            OnnxGeneratedTokens secondGenerated = workspace.resetGeneratedTokens(1);
            workspace.finishRequest();
            workspace.close();

            assertSame(firstHistory, secondHistory);
            assertSame(firstGenerated, secondGenerated);
            assertEquals(1, secondHistory.size());
            assertEquals(7, secondHistory.last());
            assertTrue(secondGenerated.isEmpty());
        }
    }

    @Test
    void rejectsUseAfterClose() {
        try (Arena arena = Arena.ofConfined()) {
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunOutputValues outputValues = OnnxRunOutputValues.allocate(1);
            OnnxTextRunWorkspace workspace = createWorkspace(
                    arena,
                    false,
                    2,
                    0,
                    new FakeInputOps(arena),
                    new FakeRunOps(arena.allocate(Long.BYTES)),
                    outputValues,
                    released);

            workspace.close();

            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 1 });
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(false, false, 1, 1, 0);
            assertThrows(IllegalStateException.class, () -> workspace.assembleInputs(tokens, 1, step));
        }
    }

    private static OnnxTextRunWorkspace createWorkspace(
            Arena arena,
            boolean hasKvInputs,
            int inputCount,
            int pastKvInputCount,
            FakeInputOps inputOps,
            FakeRunOps runOps,
            OnnxRunOutputValues outputValues,
            List<MemorySegment> released) {
        OnnxRepeatedInputValue emptyPastKvInput = OnnxRepeatedInputValue.create(
                pastKvInputCount,
                () -> arena.allocate(Long.BYTES),
                released::add);
        OnnxTextInputAssembler inputAssembler = createInputAssembler(
                arena,
                inputOps,
                inputCount,
                false,
                hasKvInputs,
                pastKvInputCount,
                emptyPastKvInput);
        return OnnxTextRunWorkspace.createForTest(
                8,
                hasKvInputs,
                OnnxPastKvState.allocate(pastKvInputCount),
                inputAssembler,
                createPreparedRun(arena, runOps, outputValues),
                outputValues,
                OnnxRunOutputLifecycle.create(
                        outputValues,
                        OnnxPastKvState.allocate(pastKvInputCount),
                        hasKvInputs,
                        released::add),
                OnnxLogitsSelector.createForTest(new FakeLogitsOps(), 4, hasKvInputs),
                emptyPastKvInput,
                released::add);
    }

    private static OnnxTextInputAssembler createInputAssembler(
            Arena arena,
            FakeInputOps ops,
            int inputCount,
            boolean hasPositionIds,
            boolean hasKvInputs,
            int pastKvInputCount,
            OnnxRepeatedInputValue emptyPastKvInput) {
        return OnnxTextInputAssembler.createForTest(
                ops,
                arena.allocate(Long.BYTES),
                arena.allocate(2L * Long.BYTES),
                arena.allocate(Long.BYTES),
                OnnxRunInputValues.allocate(inputCount, hasPositionIds ? 3 : 2),
                OnnxInputIdsScratch.allocate(arena, 8),
                OnnxAttentionMaskScratch.allocate(arena, 8),
                hasPositionIds ? OnnxPositionIdsScratch.allocate(arena, 8) : null,
                hasKvInputs,
                pastKvInputCount,
                emptyPastKvInput);
    }

    private static OnnxPreparedRun createPreparedRun(
            Arena arena,
            FakeRunOps ops,
            OnnxRunOutputValues outputValues) {
        return OnnxPreparedRun.createForTest(
                ops,
                arena.allocate(Long.BYTES),
                MemorySegment.NULL,
                arena.allocate(Long.BYTES),
                arena.allocate(Long.BYTES),
                arena.allocate(Long.BYTES),
                arena.allocate(Long.BYTES),
                outputValues);
    }

    private static InferenceRequest profiledRequest() {
        return InferenceRequest.builder()
                .model("onnx-test")
                .message(Message.user("hello"))
                .parameter("onnx_profile", true)
                .build();
    }

    private static final class FakeInputOps implements OnnxTextInputAssembler.Ops {
        private final Arena arena;
        private final List<MemorySegment> createdTensors = new ArrayList<>();

        private FakeInputOps(Arena arena) {
            this.arena = arena;
        }

        @Override
        public void writeShape2d(MemorySegment shapeBuffer, long firstDim, long secondDim) {
            // Shape contents are covered by OnnxTextInputAssemblerTest.
        }

        @Override
        public MemorySegment createInt64Tensor(
                MemorySegment memInfo,
                OnnxTensorDataView view,
                MemorySegment shapeBuffer,
                MemorySegment valueOutPointer) {
            MemorySegment tensor = arena.allocate(Long.BYTES);
            createdTensors.add(tensor);
            return tensor;
        }
    }

    private static final class FakeRunOps implements OnnxPreparedRun.Ops {
        private final MemorySegment[] outputs;

        private FakeRunOps(MemorySegment... outputs) {
            this.outputs = outputs;
        }

        @Override
        public void writePointerArray(MemorySegment pointerArray, MemorySegment[] values, int count) {
            // Pointer packing is covered by OnnxPreparedRunTest.
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
            System.arraycopy(outputs, 0, outputValues, 0, Math.min(outputs.length, outputValues.length));
        }
    }

    private static final class FakeLogitsOps implements OnnxLogitsSelector.Ops {
        @Override
        public OnnxRuntimeBinding.LogitsTailPlan createTailPlan(
                MemorySegment logits,
                int fallbackWidth,
                long sequenceLength) {
            return new OnnxRuntimeBinding.LogitsTailPlan(
                    OnnxRuntimeBinding.ONNX_TENSOR_FLOAT,
                    fallbackWidth,
                    true);
        }

        @Override
        public int argmaxLast(MemorySegment logits, int fallbackWidth) {
            return 102;
        }

        @Override
        public int argmaxLast(
                MemorySegment logits,
                OnnxRuntimeBinding.LogitsTailPlan tailPlan,
                long sequenceLength) {
            return 100 + (int) sequenceLength;
        }
    }
}
