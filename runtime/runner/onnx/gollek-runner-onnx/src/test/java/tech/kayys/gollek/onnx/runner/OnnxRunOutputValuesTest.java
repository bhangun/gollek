package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxRunOutputValuesTest {

    @Test
    void exposesLogitsAfterCompletedRun() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment logits = arena.allocate(8);
            OnnxRunOutputValues outputs = OnnxRunOutputValues.allocate(1);

            outputs.prepareForRun();
            outputs.values()[0] = logits;
            outputs.markRunCompleted();

            assertSame(logits, outputs.logits());
        }
    }

    @Test
    void releasesLogitsAndUncapturedPresentOutputsOnFailureCleanup() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment logits = arena.allocate(8);
            MemorySegment key = arena.allocate(8);
            MemorySegment value = arena.allocate(8);
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunOutputValues outputs = OnnxRunOutputValues.allocate(3);

            outputs.prepareForRun();
            outputs.values()[0] = logits;
            outputs.values()[1] = key;
            outputs.values()[2] = value;
            outputs.markRunCompleted();

            outputs.releaseUncaptured(released::add);
            outputs.releaseUncaptured(released::add);

            assertEquals(List.of(logits, key, value), released);
        }
    }

    @Test
    void capturedPresentOutputsAreReleasedByPastKvState() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment logits = arena.allocate(8);
            MemorySegment key = arena.allocate(8);
            MemorySegment value = arena.allocate(8);
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunOutputValues outputs = OnnxRunOutputValues.allocate(3);
            OnnxPastKvState past = OnnxPastKvState.allocate(2);

            outputs.prepareForRun();
            outputs.values()[0] = logits;
            outputs.values()[1] = key;
            outputs.values()[2] = value;
            outputs.markRunCompleted();

            outputs.capturePresentTo(past, true, released::add);
            outputs.releaseLogits(released::add);
            outputs.releaseUncaptured(released::add);

            assertEquals(List.of(logits), released);

            past.releaseCurrent(released::add);

            assertEquals(List.of(logits, key, value), released);
        }
    }

    @Test
    void prepareForRunBlocksAccessWithoutRewritingPointerArray() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment previous = arena.allocate(8);
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunOutputValues outputs = OnnxRunOutputValues.allocate(1);

            outputs.values()[0] = previous;
            outputs.markRunCompleted();
            outputs.prepareForRun();
            outputs.releaseUncaptured(released::add);

            assertSame(previous, outputs.values()[0]);
            assertThrows(IllegalStateException.class, outputs::logits);
            assertEquals(List.of(), released);
        }
    }

    @Test
    void validatesOutputCount() {
        assertThrows(IllegalArgumentException.class, () -> OnnxRunOutputValues.allocate(0));
    }
}
