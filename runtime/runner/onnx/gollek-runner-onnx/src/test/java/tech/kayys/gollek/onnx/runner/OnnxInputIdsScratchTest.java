package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

class OnnxInputIdsScratchTest {

    @Test
    void reusesGrowingPrefixForStatelessDecode() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 10, 20 });

        try (Arena arena = Arena.ofConfined()) {
            OnnxInputIdsScratch scratch = OnnxInputIdsScratch.allocate(arena, 4);

            MemorySegment first = scratch.prefix(history, history.size());

            assertEquals(2L * Long.BYTES, first.byteSize());
            assertEquals(10L, first.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(20L, first.getAtIndex(ValueLayout.JAVA_LONG, 1));
            assertEquals(2, scratch.filledPrefix());

            history.append(30);
            MemorySegment second = scratch.prefix(history, history.size());

            assertEquals(3L * Long.BYTES, second.byteSize());
            assertEquals(10L, second.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(20L, second.getAtIndex(ValueLayout.JAVA_LONG, 1));
            assertEquals(30L, second.getAtIndex(ValueLayout.JAVA_LONG, 2));
            assertEquals(3, scratch.filledPrefix());
        }
    }

    @Test
    void returnsDedicatedLatestTokenSlotForKvDecode() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 7, 8 });

        try (Arena arena = Arena.ofConfined()) {
            OnnxInputIdsScratch scratch = OnnxInputIdsScratch.allocate(arena, 4);

            scratch.prefix(history, history.size());
            history.append(9);
            MemorySegment latest = scratch.last(history);

            assertEquals(Long.BYTES, latest.byteSize());
            assertEquals(9L, latest.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(2, scratch.filledPrefix());
        }
    }

    @Test
    void prefixViewKeepsStableBackingBufferWithLogicalLength() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 10, 20 });

        try (Arena arena = Arena.ofConfined()) {
            OnnxInputIdsScratch scratch = OnnxInputIdsScratch.allocate(arena, 4);

            OnnxTensorDataView first = scratch.prefixView(history, history.size());
            history.append(30);
            OnnxTensorDataView second = scratch.prefixView(history, history.size());

            assertSame(first.data(), second.data());
            assertEquals(4L * Long.BYTES, second.data().byteSize());
            assertEquals(3L * Long.BYTES, second.byteLength());
            assertEquals(30L, second.data().getAtIndex(ValueLayout.JAVA_LONG, 2));
        }
    }

    @Test
    void latestTokenSlotDoesNotRequirePrefixCapacity() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 7, 8, 9 });

        try (Arena arena = Arena.ofConfined()) {
            OnnxInputIdsScratch scratch = OnnxInputIdsScratch.allocate(arena, 1);

            MemorySegment latest = scratch.last(history);

            assertEquals(Long.BYTES, latest.byteSize());
            assertEquals(9L, latest.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(0, scratch.filledPrefix());
        }
    }

    @Test
    void shorterPrefixViewsDoNotShrinkFilledPrefix() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 1, 2, 3 });

        try (Arena arena = Arena.ofConfined()) {
            OnnxInputIdsScratch scratch = OnnxInputIdsScratch.allocate(arena, 3);

            scratch.prefix(history, 3);
            MemorySegment shorter = scratch.prefix(history, 1);

            assertEquals(Long.BYTES, shorter.byteSize());
            assertEquals(1L, shorter.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(3, scratch.filledPrefix());
        }
    }

    @Test
    void resetAllowsDifferentPromptPrefixToOverwriteCachedValues() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxInputIdsScratch scratch = OnnxInputIdsScratch.allocate(arena, 3);

            scratch.prefix(OnnxTokenHistory.from(new int[] { 1, 2, 3 }), 3);
            scratch.reset();
            MemorySegment prefix = scratch.prefix(OnnxTokenHistory.from(new int[] { 9, 8 }), 2);

            assertEquals(2, scratch.filledPrefix());
            assertEquals(9L, prefix.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(8L, prefix.getAtIndex(ValueLayout.JAVA_LONG, 1));
        }
    }

    @Test
    void rejectsInvalidCapacityAndRanges() {
        assertThrows(IllegalArgumentException.class,
                () -> OnnxInputIdsScratch.allocate(Arena.ofAuto(), 0));

        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 1, 2 });
        try (Arena arena = Arena.ofConfined()) {
            OnnxInputIdsScratch scratch = OnnxInputIdsScratch.allocate(arena, 1);

            assertThrows(IllegalArgumentException.class, () -> scratch.prefix(history, 0));
            assertThrows(IllegalArgumentException.class, () -> scratch.prefix(history, 2));
            assertThrows(IllegalArgumentException.class, () -> scratch.prefix(history, 3));
        }

        try (Arena arena = Arena.ofConfined()) {
            OnnxInputIdsScratch scratch = OnnxInputIdsScratch.allocate(arena, 1);

            assertThrows(IllegalStateException.class,
                    () -> scratch.last(OnnxTokenHistory.from(new int[0])));
        }
    }
}
