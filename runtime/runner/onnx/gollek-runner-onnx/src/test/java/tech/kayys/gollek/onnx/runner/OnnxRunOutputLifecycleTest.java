package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxRunOutputLifecycleTest {

    @Test
    void capturesPresentKvAndReleasesOnlyLogitsImmediately() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment logits = arena.allocate(Long.BYTES);
            MemorySegment key = arena.allocate(Long.BYTES);
            MemorySegment value = arena.allocate(Long.BYTES);
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunOutputValues outputs = completedOutputs(logits, key, value);
            OnnxPastKvState past = OnnxPastKvState.allocate(2);
            OnnxRunOutputLifecycle lifecycle = OnnxRunOutputLifecycle.create(outputs, past, true, released::add);

            lifecycle.capturePresentAndReleaseLogits();

            assertEquals(List.of(logits), released);

            lifecycle.releaseUncapturedAndCurrentPastKv();

            assertEquals(List.of(logits, key, value), released);
        }
    }

    @Test
    void releasesOnlyLogitsWhenKvCaptureIsDisabled() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment logits = arena.allocate(Long.BYTES);
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunOutputValues outputs = completedOutputs(logits);
            OnnxRunOutputLifecycle lifecycle = OnnxRunOutputLifecycle.create(
                    outputs,
                    OnnxPastKvState.allocate(0),
                    false,
                    released::add);

            lifecycle.capturePresentAndReleaseLogits();
            lifecycle.releaseUncapturedAndCurrentPastKv();

            assertEquals(List.of(logits), released);
        }
    }

    @Test
    void cleanupIsSafeWhenRunWasNotCompleted() {
        try (Arena arena = Arena.ofConfined()) {
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunOutputValues outputs = OnnxRunOutputValues.allocate(1);
            OnnxRunOutputLifecycle lifecycle = OnnxRunOutputLifecycle.create(
                    outputs,
                    OnnxPastKvState.allocate(0),
                    false,
                    released::add);

            lifecycle.releaseUncapturedAndCurrentPastKv();

            assertEquals(List.of(), released);
        }
    }

    @Test
    void validatesRequiredArguments() {
        OnnxRunOutputValues outputs = OnnxRunOutputValues.allocate(1);
        OnnxPastKvState past = OnnxPastKvState.allocate(0);

        assertThrows(NullPointerException.class,
                () -> OnnxRunOutputLifecycle.create(null, past, false, ignored -> {
                }));
        assertThrows(NullPointerException.class,
                () -> OnnxRunOutputLifecycle.create(outputs, null, false, ignored -> {
                }));
        assertThrows(NullPointerException.class,
                () -> OnnxRunOutputLifecycle.create(outputs, past, false, null));
    }

    private static OnnxRunOutputValues completedOutputs(MemorySegment... values) {
        OnnxRunOutputValues outputs = OnnxRunOutputValues.allocate(values.length);
        outputs.prepareForRun();
        System.arraycopy(values, 0, outputs.values(), 0, values.length);
        outputs.markRunCompleted();
        return outputs;
    }
}
