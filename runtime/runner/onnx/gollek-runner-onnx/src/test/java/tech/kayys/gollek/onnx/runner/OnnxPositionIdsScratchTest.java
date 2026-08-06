package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

class OnnxPositionIdsScratchTest {

    @Test
    void growsPositionIdsIncrementally() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxPositionIdsScratch scratch = OnnxPositionIdsScratch.allocate(arena, 6);

            MemorySegment prefill = scratch.positions(0, 3);

            assertEquals(3L * Long.BYTES, prefill.byteSize());
            assertEquals(0L, prefill.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(1L, prefill.getAtIndex(ValueLayout.JAVA_LONG, 1));
            assertEquals(2L, prefill.getAtIndex(ValueLayout.JAVA_LONG, 2));
            assertEquals(3L, scratch.filledPositions());

            MemorySegment decode = scratch.positions(3, 2);

            assertEquals(2L * Long.BYTES, decode.byteSize());
            assertEquals(3L, decode.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(4L, decode.getAtIndex(ValueLayout.JAVA_LONG, 1));
            assertEquals(5L, scratch.filledPositions());
        }
    }

    @Test
    void fullContextViewsCanRestartAtZeroForStatelessDecode() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxPositionIdsScratch scratch = OnnxPositionIdsScratch.allocate(arena, 5);

            scratch.positions(0, 4);
            MemorySegment restarted = scratch.positions(0, 2);

            assertEquals(2L * Long.BYTES, restarted.byteSize());
            assertEquals(0L, restarted.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(1L, restarted.getAtIndex(ValueLayout.JAVA_LONG, 1));
            assertEquals(4L, scratch.filledPositions());
        }
    }

    @Test
    void zeroStartViewKeepsStableBackingBufferWithLogicalLength() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxPositionIdsScratch scratch = OnnxPositionIdsScratch.allocate(arena, 5);

            OnnxTensorDataView first = scratch.positionsView(0, 2);
            OnnxTensorDataView second = scratch.positionsView(0, 4);

            assertSame(first.data(), second.data());
            assertEquals(5L * Long.BYTES, second.data().byteSize());
            assertEquals(4L * Long.BYTES, second.byteLength());
            assertEquals(3L, second.data().getAtIndex(ValueLayout.JAVA_LONG, 3));
        }
    }

    @Test
    void singleDecodePositionUsesDedicatedScalarSlot() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxPositionIdsScratch scratch = OnnxPositionIdsScratch.allocate(arena, 6);

            scratch.positionsView(0, 3);
            OnnxTensorDataView decode = scratch.positionsView(3, 1);
            OnnxTensorDataView nextDecode = scratch.positionsView(4, 1);

            assertSame(decode.data(), nextDecode.data());
            assertEquals(Long.BYTES, decode.byteLength());
            assertEquals(4L, nextDecode.data().getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(3L, scratch.filledPositions());
        }
    }

    @Test
    void rejectsInvalidCapacityAndRanges() {
        assertThrows(IllegalArgumentException.class,
                () -> OnnxPositionIdsScratch.allocate(Arena.ofAuto(), 0));

        try (Arena arena = Arena.ofConfined()) {
            OnnxPositionIdsScratch scratch = OnnxPositionIdsScratch.allocate(arena, 2);

            assertThrows(IllegalArgumentException.class, () -> scratch.positions(-1, 1));
            assertThrows(IllegalArgumentException.class, () -> scratch.positions(0, 0));
            assertThrows(IllegalArgumentException.class, () -> scratch.positions(2, 1));
            assertThrows(IllegalArgumentException.class, () -> scratch.positions(1, 2));
        }
    }
}
