package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

class OnnxAttentionMaskScratchTest {

    @Test
    void growsAllOnesMaskIncrementally() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxAttentionMaskScratch scratch = OnnxAttentionMaskScratch.allocate(arena, 5);

            MemorySegment first = scratch.ones(2);

            assertEquals(2L * Long.BYTES, first.byteSize());
            assertEquals(1L, first.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(1L, first.getAtIndex(ValueLayout.JAVA_LONG, 1));
            assertEquals(2L, scratch.filledOnes());

            MemorySegment second = scratch.ones(4);

            assertEquals(4L * Long.BYTES, second.byteSize());
            assertEquals(1L, second.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(1L, second.getAtIndex(ValueLayout.JAVA_LONG, 1));
            assertEquals(1L, second.getAtIndex(ValueLayout.JAVA_LONG, 2));
            assertEquals(1L, second.getAtIndex(ValueLayout.JAVA_LONG, 3));
            assertEquals(4L, scratch.filledOnes());
        }
    }

    @Test
    void reusingShorterViewDoesNotShrinkFilledPrefix() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxAttentionMaskScratch scratch = OnnxAttentionMaskScratch.allocate(arena, 4);

            scratch.ones(4);
            MemorySegment shorter = scratch.ones(1);

            assertEquals(Long.BYTES, shorter.byteSize());
            assertEquals(1L, shorter.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(4L, scratch.filledOnes());
        }
    }

    @Test
    void onesViewKeepsStableBackingBufferWithLogicalLength() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxAttentionMaskScratch scratch = OnnxAttentionMaskScratch.allocate(arena, 4);

            OnnxTensorDataView first = scratch.onesView(2);
            OnnxTensorDataView second = scratch.onesView(4);

            assertSame(first.data(), second.data());
            assertEquals(4L * Long.BYTES, second.data().byteSize());
            assertEquals(4L * Long.BYTES, second.byteLength());
            assertEquals(1L, second.data().getAtIndex(ValueLayout.JAVA_LONG, 3));
        }
    }

    @Test
    void rejectsInvalidCapacityAndLengths() {
        assertThrows(IllegalArgumentException.class,
                () -> OnnxAttentionMaskScratch.allocate(Arena.ofAuto(), 0));

        try (Arena arena = Arena.ofConfined()) {
            OnnxAttentionMaskScratch scratch = OnnxAttentionMaskScratch.allocate(arena, 2);

            assertThrows(IllegalArgumentException.class, () -> scratch.ones(0));
            assertThrows(IllegalArgumentException.class, () -> scratch.ones(3));
        }
    }
}
