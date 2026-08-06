package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxPastKvStateTest {

    @Test
    void rotatesReusableBuffersAndReleasesPreviousCurrent() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment logitsA = arena.allocate(Long.BYTES);
            MemorySegment keyA = arena.allocate(Long.BYTES);
            MemorySegment valueA = arena.allocate(Long.BYTES);
            MemorySegment logitsB = arena.allocate(Long.BYTES);
            MemorySegment keyB = arena.allocate(Long.BYTES);
            MemorySegment valueB = arena.allocate(Long.BYTES);
            List<MemorySegment> released = new ArrayList<>();

            OnnxPastKvState state = OnnxPastKvState.allocate(2);
            state.capturePresentOutputs(
                    new MemorySegment[] { logitsA, keyA, valueA },
                    1,
                    released::add);

            assertTrue(state.hasCurrent());
            assertArrayEquals(new MemorySegment[] { keyA, valueA }, state.current());
            assertTrue(released.isEmpty());

            state.capturePresentOutputs(
                    new MemorySegment[] { logitsB, keyB, valueB },
                    1,
                    released::add);

            assertArrayEquals(new MemorySegment[] { keyB, valueB }, state.current());
            assertIterableEquals(List.of(keyA, valueA), released);

            state.releaseCurrent(released::add);

            assertFalse(state.hasCurrent());
            assertIterableEquals(List.of(keyA, valueA, keyB, valueB), released);
        }
    }

    @Test
    void zeroValueStateIsNoop() {
        OnnxPastKvState state = OnnxPastKvState.allocate(0);
        List<MemorySegment> released = new ArrayList<>();

        state.capturePresentOutputs(new MemorySegment[0], 0, released::add);
        state.releaseCurrent(released::add);

        assertFalse(state.hasCurrent());
        assertTrue(released.isEmpty());
    }

    @Test
    void rejectsInvalidCountsAndOutputWindows() {
        assertThrows(IllegalArgumentException.class, () -> OnnxPastKvState.allocate(-1));

        OnnxPastKvState state = OnnxPastKvState.allocate(2);
        assertThrows(IllegalArgumentException.class,
                () -> state.capturePresentOutputs(new MemorySegment[2], 1, ignored -> {
                }));
        assertThrows(IllegalArgumentException.class,
                () -> state.capturePresentOutputs(new MemorySegment[3], -1, ignored -> {
                }));
    }
}
