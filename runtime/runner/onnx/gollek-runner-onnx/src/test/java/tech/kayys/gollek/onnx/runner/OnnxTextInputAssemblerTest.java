package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxTextInputAssemblerTest {

    @Test
    void assemblesKvPrefillWithPositionIdsAndEmptyPastKv() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            MemorySegment repeatedKv = arena.allocate(Long.BYTES);
            OnnxRepeatedInputValue emptyKv = OnnxRepeatedInputValue.create(2, () -> repeatedKv, ignored -> {
            });
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 5, true, 2, emptyKv);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 10, 20, 30 });
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(true, false, 3, 3, 0);

            OnnxRunInputValues values = assembler.assemble(tokens, 3, step, OnnxPastKvState.allocate(2));

            assertArrayEquals(new MemorySegment[] {
                    ops.createdTensors.get(0),
                    ops.createdTensors.get(1),
                    ops.createdTensors.get(2),
                    repeatedKv,
                    repeatedKv
            }, values.completeValues());
            assertEquals(List.of(3L, 3L, 3L), ops.shapeLengths);
            assertEquals(List.of(3L * Long.BYTES, 3L * Long.BYTES, 3L * Long.BYTES), ops.tensorByteLengths);
            assertEquals(5, values.count());
        }
    }

    @Test
    void assemblesKvDecodeWithBorrowedPastKvAndScalarPosition() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            OnnxRepeatedInputValue emptyKv = OnnxRepeatedInputValue.create(2, () -> arena.allocate(Long.BYTES),
                    ignored -> {
                    });
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 5, true, 2, emptyKv);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 10, 20, 30 });
            MemorySegment key = arena.allocate(Long.BYTES);
            MemorySegment value = arena.allocate(Long.BYTES);
            OnnxPastKvState kv = OnnxPastKvState.allocate(2);
            kv.capturePresentOutputs(new MemorySegment[] { key, value }, 0, ignored -> {
            });
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(true, true, 2, 3, 2);

            OnnxRunInputValues inputs = assembler.assemble(tokens, 2, step, kv);

            assertArrayEquals(new MemorySegment[] {
                    ops.createdTensors.get(0),
                    ops.createdTensors.get(1),
                    ops.createdTensors.get(2),
                    key,
                    value
            }, inputs.completeValues());
            assertEquals(List.of(1L, 3L, 1L), ops.shapeLengths);
            assertEquals(List.of(1L * Long.BYTES, 3L * Long.BYTES, 1L * Long.BYTES), ops.tensorByteLengths);
            assertEquals(30L, ops.tensorFirstValues.get(0));
            assertEquals(1L, ops.tensorFirstValues.get(1));
            assertEquals(2L, ops.tensorFirstValues.get(2));
        }
    }

    @Test
    void reusesScalarKvDecodeTensorsAcrossSteps() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            OnnxRepeatedInputValue emptyKv = OnnxRepeatedInputValue.create(2, () -> arena.allocate(Long.BYTES),
                    ignored -> {
                    });
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 5, true, 2, emptyKv);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 10, 20, 30 });
            MemorySegment key = arena.allocate(Long.BYTES);
            MemorySegment value = arena.allocate(Long.BYTES);
            OnnxPastKvState kv = OnnxPastKvState.allocate(2);
            kv.capturePresentOutputs(new MemorySegment[] { key, value }, 0, ignored -> {
            });
            OnnxTextDecodeStep firstStep = OnnxTextDecodeStep.plan(true, true, 2, 3, 2);
            OnnxRunInputValues first = assembler.assemble(tokens, 2, firstStep, kv);
            MemorySegment firstInputIds = first.completeValues()[0];
            MemorySegment firstAttention = first.completeValues()[1];
            MemorySegment firstPositionIds = first.completeValues()[2];

            tokens.append(40);
            OnnxTextDecodeStep secondStep = OnnxTextDecodeStep.plan(true, true, 2, 4, 3);
            OnnxRunInputValues second = assembler.assemble(tokens, 2, secondStep, kv);

            assertSame(firstInputIds, second.completeValues()[0]);
            assertNotSame(firstAttention, second.completeValues()[1]);
            assertSame(firstPositionIds, second.completeValues()[2]);
            assertEquals(4, ops.createdTensors.size());
            assertEquals(List.of(1L, 3L, 1L, 4L), ops.shapeLengths);
            assertEquals(List.of(
                    1L * Long.BYTES,
                    3L * Long.BYTES,
                    1L * Long.BYTES,
                    4L * Long.BYTES), ops.tensorByteLengths);
            assertEquals(new OnnxInputTensorCacheStats(1, 1, 1, 1, 0, 2, 0, 0), assembler.cacheStats());
        }
    }

    @Test
    void reusesAttentionMaskTensorByLengthAcrossRequests() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 2, false, 0, null);
            OnnxTokenHistory firstTokens = OnnxTokenHistory.from(new int[] { 10, 20, 30 });
            OnnxTextDecodeStep firstStep = OnnxTextDecodeStep.plan(false, false, 2, 3, 0);
            OnnxRunInputValues first = assembler.assemble(firstTokens, 2, firstStep, OnnxPastKvState.allocate(0));
            MemorySegment firstInputIds = first.completeValues()[0];
            MemorySegment firstAttention = first.completeValues()[1];

            assembler.resetForRequest();
            OnnxTokenHistory secondTokens = OnnxTokenHistory.from(new int[] { 40, 50, 60 });
            OnnxTextDecodeStep secondStep = OnnxTextDecodeStep.plan(false, false, 2, 3, 0);
            OnnxRunInputValues second = assembler.assemble(secondTokens, 2, secondStep, OnnxPastKvState.allocate(0));

            assertSame(firstInputIds, second.completeValues()[0]);
            assertSame(firstAttention, second.completeValues()[1]);
            assertEquals(2, ops.createdTensors.size());
            assertEquals(List.of(3L, 3L), ops.shapeLengths);
            assertEquals(new OnnxInputTensorCacheStats(0, 0, 0, 0, 1, 0, 1, 0), assembler.cacheStats());
        }
    }

    @Test
    void reusesPrefixInputIdsTensorByLengthAcrossRequests() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            MemorySegment repeatedKv = arena.allocate(Long.BYTES);
            OnnxRepeatedInputValue emptyKv = OnnxRepeatedInputValue.create(2, () -> repeatedKv, ignored -> {
            });
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 5, true, 2, emptyKv);
            OnnxTokenHistory firstTokens = OnnxTokenHistory.from(new int[] { 10, 20, 30 });
            OnnxTextDecodeStep firstStep = OnnxTextDecodeStep.plan(true, false, 3, 3, 0);
            OnnxRunInputValues first = assembler.assemble(firstTokens, 3, firstStep, OnnxPastKvState.allocate(2));
            MemorySegment firstInputIds = first.completeValues()[0];
            MemorySegment firstAttention = first.completeValues()[1];
            MemorySegment firstPositionIds = first.completeValues()[2];

            assembler.resetForRequest();
            OnnxTokenHistory secondTokens = OnnxTokenHistory.from(new int[] { 40, 50, 60 });
            OnnxTextDecodeStep secondStep = OnnxTextDecodeStep.plan(true, false, 3, 3, 0);
            OnnxRunInputValues second = assembler.assemble(secondTokens, 3, secondStep, OnnxPastKvState.allocate(2));

            assertSame(firstInputIds, second.completeValues()[0]);
            assertSame(firstAttention, second.completeValues()[1]);
            assertSame(firstPositionIds, second.completeValues()[2]);
            assertEquals(3, ops.createdTensors.size());
            assertEquals(List.of(3L, 3L, 3L), ops.shapeLengths);
            assertEquals(new OnnxInputTensorCacheStats(0, 0, 0, 0, 1, 0, 1, 0, 1, 0),
                    assembler.cacheStats());
        }
    }

    @Test
    void reusesPositionIdRangesByStartAndLength() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 3, true, 0, null);
            OnnxTokenHistory firstTokens = OnnxTokenHistory.from(new int[] { 10, 20, 30 });
            OnnxTextDecodeStep firstStep = new OnnxTextDecodeStep(false, 3, 3, 0);
            OnnxRunInputValues first = assembler.assemble(firstTokens, 2, firstStep, OnnxPastKvState.allocate(0));
            MemorySegment firstInputIds = first.completeValues()[0];
            MemorySegment firstAttention = first.completeValues()[1];
            MemorySegment firstPositionIds = first.completeValues()[2];

            assembler.resetForRequest();
            OnnxTokenHistory secondTokens = OnnxTokenHistory.from(new int[] { 40, 50, 60 });
            OnnxTextDecodeStep sameRange = new OnnxTextDecodeStep(false, 3, 3, 0);
            OnnxRunInputValues second = assembler.assemble(secondTokens, 2, sameRange, OnnxPastKvState.allocate(0));
            MemorySegment secondPositionIds = second.completeValues()[2];

            OnnxTextDecodeStep shiftedRange = new OnnxTextDecodeStep(false, 3, 3, 1);
            OnnxRunInputValues shifted = assembler.assemble(secondTokens, 2, shiftedRange, OnnxPastKvState.allocate(0));

            assertSame(firstInputIds, second.completeValues()[0]);
            assertSame(firstAttention, second.completeValues()[1]);
            assertSame(firstPositionIds, secondPositionIds);
            assertNotSame(firstPositionIds, shifted.completeValues()[2]);
            assertEquals(4, ops.createdTensors.size());
            assertEquals(List.of(3L, 3L, 3L, 3L), ops.shapeLengths);
            assertEquals(new OnnxInputTensorCacheStats(0, 0, 0, 0, 2, 0, 2, 0, 1, 1),
                    assembler.cacheStats());
        }
    }

    @Test
    void evictsOldestVariableLengthTensorsWhenCacheLimitIsReached() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            List<MemorySegment> released = new ArrayList<>();
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 2, false, 0, null, 1, released);
            OnnxTokenHistory firstTokens = OnnxTokenHistory.from(new int[] { 10, 20 });
            OnnxTextDecodeStep firstStep = OnnxTextDecodeStep.plan(false, false, 2, 2, 0);
            OnnxRunInputValues first = assembler.assemble(firstTokens, 2, firstStep, OnnxPastKvState.allocate(0));
            MemorySegment firstInputIds = first.completeValues()[0];
            MemorySegment firstAttention = first.completeValues()[1];

            assembler.resetForRequest();
            OnnxTokenHistory secondTokens = OnnxTokenHistory.from(new int[] { 10, 20, 30 });
            OnnxTextDecodeStep secondStep = OnnxTextDecodeStep.plan(false, false, 2, 3, 0);
            OnnxRunInputValues second = assembler.assemble(secondTokens, 2, secondStep, OnnxPastKvState.allocate(0));

            assertNotSame(firstInputIds, second.completeValues()[0]);
            assertNotSame(firstAttention, second.completeValues()[1]);
            assertEquals(List.of(firstInputIds, firstAttention), released);
            assertEquals(new OnnxInputTensorCacheStats(0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2),
                    assembler.cacheStats());
        }
    }

    @Test
    void cacheSizeZeroReturnsOwnedVariableLengthTensors() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            List<MemorySegment> released = new ArrayList<>();
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 2, false, 0, null, 0, released);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 10, 20 });
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(false, false, 2, 2, 0);

            OnnxRunInputValues values = assembler.assemble(tokens, 2, step, OnnxPastKvState.allocate(0));
            OnnxInputTensorCacheStats stats = assembler.cacheStats();
            values.releaseOwned(released::add);
            assembler.releaseCachedInputs(released::add);

            assertEquals(List.of(values.completeValues()[0], values.completeValues()[1]), released);
            assertEquals(new OnnxInputTensorCacheStats(0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0), stats);
        }
    }

    @Test
    void releasesCachedInputTensorsOnce() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            OnnxRepeatedInputValue emptyKv = OnnxRepeatedInputValue.create(2, () -> arena.allocate(Long.BYTES),
                    ignored -> {
                    });
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 5, true, 2, emptyKv);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 10, 20, 30 });
            MemorySegment key = arena.allocate(Long.BYTES);
            MemorySegment value = arena.allocate(Long.BYTES);
            OnnxPastKvState kv = OnnxPastKvState.allocate(2);
            kv.capturePresentOutputs(new MemorySegment[] { key, value }, 0, ignored -> {
            });
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(true, true, 2, 3, 2);
            OnnxRunInputValues values = assembler.assemble(tokens, 2, step, kv);
            MemorySegment inputIds = values.completeValues()[0];
            MemorySegment attentionMask = values.completeValues()[1];
            MemorySegment positionIds = values.completeValues()[2];
            List<MemorySegment> released = new ArrayList<>();

            assembler.releaseCachedInputs(released::add);
            assembler.releaseCachedInputs(released::add);

            assertEquals(List.of(inputIds, positionIds, attentionMask), released);
        }
    }

    @Test
    void assemblesStatelessInputsWithoutPositionOrKv() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 2, false, 0, null);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 10, 20 });
            tokens.append(30);
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(false, false, 2, 3, 99);

            OnnxRunInputValues values = assembler.assemble(tokens, 2, step, OnnxPastKvState.allocate(0));

            assertArrayEquals(new MemorySegment[] { ops.createdTensors.get(0), ops.createdTensors.get(1) },
                    values.completeValues());
            assertEquals(List.of(3L, 3L), ops.shapeLengths);
            assertEquals(List.of(3L * Long.BYTES, 3L * Long.BYTES), ops.tensorByteLengths);
            assertEquals(10L, ops.tensorFirstValues.get(0));
        }
    }

    @Test
    void validatesConstructionAndKvState() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);

            assertThrows(NullPointerException.class,
                    () -> createAssembler(arena, ops, 2, false, 1, null));
            assertThrows(IllegalArgumentException.class,
                    () -> OnnxTextInputAssembler.createForTest(
                            ops,
                            arena.allocate(Long.BYTES),
                            arena.allocate(2L * Long.BYTES),
                            arena.allocate(Long.BYTES),
                            OnnxRunInputValues.allocate(2, 2),
                            OnnxInputIdsScratch.allocate(arena, 2),
                            OnnxAttentionMaskScratch.allocate(arena, 2),
                            null,
                            false,
                            -1,
                            null));

            OnnxRepeatedInputValue emptyKv = OnnxRepeatedInputValue.create(1, () -> arena.allocate(Long.BYTES),
                    ignored -> {
                    });
            OnnxTextInputAssembler assembler = createAssembler(arena, ops, 3, false, 1, emptyKv);
            OnnxTokenHistory tokens = OnnxTokenHistory.from(new int[] { 1 });
            OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(true, false, 1, 1, 0);

            assertThrows(NullPointerException.class, () -> assembler.assemble(tokens, 1, step, null));
        }
    }

    private static OnnxTextInputAssembler createAssembler(
            Arena arena,
            FakeOps ops,
            int inputCount,
            boolean hasPositionIds,
            int pastKvInputCount,
            OnnxRepeatedInputValue emptyPastKvInput) {
        int ownedInputCount = hasPositionIds ? 3 : 2;
        return OnnxTextInputAssembler.createForTest(
                ops,
                arena.allocate(Long.BYTES),
                arena.allocate(2L * Long.BYTES),
                arena.allocate(Long.BYTES),
                OnnxRunInputValues.allocate(inputCount, ownedInputCount),
                OnnxInputIdsScratch.allocate(arena, 8),
                OnnxAttentionMaskScratch.allocate(arena, 8),
                hasPositionIds ? OnnxPositionIdsScratch.allocate(arena, 8) : null,
                pastKvInputCount > 0,
                pastKvInputCount,
                emptyPastKvInput);
    }

    private static OnnxTextInputAssembler createAssembler(
            Arena arena,
            FakeOps ops,
            int inputCount,
            boolean hasPositionIds,
            int pastKvInputCount,
            OnnxRepeatedInputValue emptyPastKvInput,
            int inputTensorCacheEntries,
            List<MemorySegment> released) {
        int ownedInputCount = hasPositionIds ? 3 : 2;
        return OnnxTextInputAssembler.createForTest(
                ops,
                arena.allocate(Long.BYTES),
                arena.allocate(2L * Long.BYTES),
                arena.allocate(Long.BYTES),
                OnnxRunInputValues.allocate(inputCount, ownedInputCount),
                OnnxInputIdsScratch.allocate(arena, 8),
                OnnxAttentionMaskScratch.allocate(arena, 8),
                hasPositionIds ? OnnxPositionIdsScratch.allocate(arena, 8) : null,
                pastKvInputCount > 0,
                pastKvInputCount,
                emptyPastKvInput,
                inputTensorCacheEntries,
                released::add);
    }

    private static final class FakeOps implements OnnxTextInputAssembler.Ops {
        private final Arena arena;
        private final List<MemorySegment> createdTensors = new ArrayList<>();
        private final List<Long> shapeLengths = new ArrayList<>();
        private final List<Long> tensorByteLengths = new ArrayList<>();
        private final List<Long> tensorFirstValues = new ArrayList<>();

        private FakeOps(Arena arena) {
            this.arena = arena;
        }

        @Override
        public void writeShape2d(MemorySegment shapeBuffer, long firstDim, long secondDim) {
            shapeBuffer.setAtIndex(ValueLayout.JAVA_LONG, 0, firstDim);
            shapeBuffer.setAtIndex(ValueLayout.JAVA_LONG, 1, secondDim);
            shapeLengths.add(secondDim);
        }

        @Override
        public MemorySegment createInt64Tensor(
                MemorySegment memInfo,
                OnnxTensorDataView view,
                MemorySegment shapeBuffer,
                MemorySegment valueOutPointer) {
            MemorySegment tensor = arena.allocate(Long.BYTES);
            createdTensors.add(tensor);
            tensorByteLengths.add(view.byteLength());
            tensorFirstValues.add(view.data().getAtIndex(ValueLayout.JAVA_LONG, 0));
            return tensor;
        }
    }
}
