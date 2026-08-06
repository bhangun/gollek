package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;

class OnnxPreparedRunTest {

    @Test
    void writesInputPointersRunsOrtAndMarksOutputsCompleted() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            OnnxRunInputValues inputValues = OnnxRunInputValues.allocate(2, 0);
            MemorySegment firstInput = arena.allocate(Long.BYTES);
            MemorySegment secondInput = arena.allocate(Long.BYTES);
            inputValues.addBorrowed(firstInput);
            inputValues.addBorrowed(secondInput);
            OnnxRunOutputValues outputValues = OnnxRunOutputValues.allocate(2);
            OnnxPreparedRun run = createRun(arena, ops, outputValues);
            OnnxInferenceProfile profile = OnnxInferenceProfile.start(profiledRequest());

            OnnxRunOutputValues result = run.execute(inputValues, profile, profile.mark(), true);

            assertSame(outputValues, result);
            assertSame(ops.logits, result.logits());
            assertArrayEquals(new MemorySegment[] { firstInput, secondInput }, ops.writtenInputValues);
            assertEquals(2, ops.writtenInputCount);
            assertEquals(2, ops.runInputCount);
            assertEquals(List.of("write", "run"), ops.events);
        }
    }

    @Test
    void incompleteInputsFailBeforePointerWriteOrRun() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            OnnxRunInputValues inputValues = OnnxRunInputValues.allocate(1, 0);
            OnnxRunOutputValues outputValues = OnnxRunOutputValues.allocate(1);
            OnnxPreparedRun run = createRun(arena, ops, outputValues);

            assertThrows(IllegalStateException.class,
                    () -> run.execute(inputValues, OnnxInferenceProfile.start(profiledRequest()), 0L, false));
            assertEquals(List.of(), ops.events);
            assertThrows(IllegalStateException.class, outputValues::logits);
        }
    }

    @Test
    void validatesArguments() {
        try (Arena arena = Arena.ofConfined()) {
            FakeOps ops = new FakeOps(arena);
            OnnxPreparedRun run = createRun(arena, ops, OnnxRunOutputValues.allocate(1));
            OnnxRunInputValues inputValues = OnnxRunInputValues.allocate(1, 0);
            inputValues.addBorrowed(MemorySegment.NULL);

            assertThrows(NullPointerException.class,
                    () -> run.execute(null, OnnxInferenceProfile.start(null), 0L, false));
            assertThrows(NullPointerException.class, () -> run.execute(inputValues, null, 0L, false));
            assertThrows(NullPointerException.class,
                    () -> OnnxPreparedRun.createForTest(
                            null,
                            arena.allocate(Long.BYTES),
                            MemorySegment.NULL,
                            arena.allocate(Long.BYTES),
                            arena.allocate(Long.BYTES),
                            arena.allocate(Long.BYTES),
                            arena.allocate(Long.BYTES),
                            OnnxRunOutputValues.allocate(1)));
        }
    }

    private static OnnxPreparedRun createRun(Arena arena, FakeOps ops, OnnxRunOutputValues outputValues) {
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

    private static final class FakeOps implements OnnxPreparedRun.Ops {
        private final List<String> events = new ArrayList<>();
        private final MemorySegment logits;
        private final MemorySegment present;
        private MemorySegment[] writtenInputValues;
        private int writtenInputCount;
        private int runInputCount;

        private FakeOps(Arena arena) {
            logits = arena.allocate(Long.BYTES);
            present = arena.allocate(Long.BYTES);
        }

        @Override
        public void writePointerArray(MemorySegment pointerArray, MemorySegment[] values, int count) {
            events.add("write");
            writtenInputValues = values.clone();
            writtenInputCount = count;
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
            events.add("run");
            runInputCount = inputCount;
            outputValues[0] = logits;
            if (outputValues.length > 1) {
                outputValues[1] = present;
            }
        }
    }
}
