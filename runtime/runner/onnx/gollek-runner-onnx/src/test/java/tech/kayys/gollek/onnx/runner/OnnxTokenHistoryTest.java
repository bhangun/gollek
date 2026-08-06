package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

class OnnxTokenHistoryTest {

    @Test
    void appendsWithoutBoxingAndWritesAllTokensAsInt64() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 10, 20 });
        history.append(30);

        assertEquals(3, history.size());
        assertEquals(30, history.last());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(3L * Long.BYTES, Long.BYTES);

            history.writeAllAsInt64(target);

            assertEquals(10L, target.getAtIndex(ValueLayout.JAVA_LONG, 0));
            assertEquals(20L, target.getAtIndex(ValueLayout.JAVA_LONG, 1));
            assertEquals(30L, target.getAtIndex(ValueLayout.JAVA_LONG, 2));
        }
    }

    @Test
    void reservesPromptPlusExpectedGeneratedTokenCapacity() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 1, 2, 3 }, 4);
        int initialCapacity = history.capacity();

        history.append(4);
        history.append(5);
        history.append(6);
        history.append(7);

        assertEquals(7, history.size());
        assertEquals(7, history.last());
        assertEquals(initialCapacity, history.capacity());
    }

    @Test
    void keepsMinimumCapacityForTinyPrompts() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 1 }, 0);

        assertEquals(1, history.size());
        assertEquals(8, history.capacity());
    }

    @Test
    void resetsFromPromptForWorkspaceReuse() {
        OnnxTokenHistory history = OnnxTokenHistory.allocate(4);

        OnnxTokenHistory first = history.resetFrom(new int[] { 1, 2, 3 }, 2);
        int firstCapacity = history.capacity();
        history.append(4);

        OnnxTokenHistory second = history.resetFrom(new int[] { 7 }, 1);

        assertSame(history, first);
        assertSame(history, second);
        assertEquals(1, history.size());
        assertEquals(7, history.last());
        assertEquals(firstCapacity, history.capacity());
    }

    @Test
    void writesLastTokenForKvDecodeStep() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 7, 8, 9 });

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(Long.BYTES, Long.BYTES);

            history.writeLastAsInt64(target);

            assertEquals(9L, target.getAtIndex(ValueLayout.JAVA_LONG, 0));
        }
    }

    @Test
    void validatesWriteBounds() {
        OnnxTokenHistory history = OnnxTokenHistory.from(new int[] { 1, 2 });

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment oneToken = arena.allocate(Long.BYTES, Long.BYTES);

            assertThrows(IllegalArgumentException.class,
                    () -> history.writePrefixAsInt64(oneToken, 2));
            assertThrows(IllegalArgumentException.class,
                    () -> history.writePrefixAsInt64(oneToken, 3));
        }

        assertThrows(IllegalStateException.class,
                () -> OnnxTokenHistory.from(new int[0]).last());
        assertThrows(IllegalArgumentException.class,
                () -> OnnxTokenHistory.from(new int[] { 1 }, -1));
        assertThrows(IllegalArgumentException.class,
                () -> OnnxTokenHistory.from(new int[] { 1 }, Integer.MAX_VALUE));
        assertThrows(IllegalArgumentException.class,
                () -> OnnxTokenHistory.allocate(-1));
        assertThrows(IllegalArgumentException.class,
                () -> OnnxTokenHistory.allocate(1).resetFrom(new int[] { 1 }, -1));
    }
}
