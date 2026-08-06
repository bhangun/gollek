package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxRunInputValuesTest {

    @Test
    void assemblesOwnedBorrowedAndRepeatedInputsInOrder() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment owned = arena.allocate(8);
            MemorySegment borrowed = arena.allocate(8);
            MemorySegment repeated = arena.allocate(8);
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunInputValues values = OnnxRunInputValues.allocate(4, 1);
            OnnxRepeatedInputValue repeatedValue = OnnxRepeatedInputValue.create(
                    2,
                    () -> repeated,
                    released::add);

            values.reset();
            values.addOwned(owned);
            values.addBorrowedAll(new MemorySegment[] { borrowed }, 1);
            values.addRepeated(repeatedValue);

            assertEquals(4, values.count());
            assertArrayEquals(new MemorySegment[] { owned, borrowed, repeated, repeated },
                    values.completeValues());

            values.releaseOwned(released::add);
            repeatedValue.release();

            assertEquals(List.of(owned, repeated), released);
        }
    }

    @Test
    void resetAllowsReusingBackingArraysAcrossRuns() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = arena.allocate(8);
            MemorySegment second = arena.allocate(8);
            OnnxRunInputValues values = OnnxRunInputValues.allocate(1, 1);

            values.addOwned(first);
            assertArrayEquals(new MemorySegment[] { first }, values.completeValues());
            values.releaseOwned(ignored -> {
            });

            values.reset();
            values.addOwned(second);

            assertArrayEquals(new MemorySegment[] { second }, values.completeValues());
        }
    }

    @Test
    void releaseOwnedIsIdempotentAndDoesNotRetainStaleOwnedValues() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = arena.allocate(8);
            MemorySegment second = arena.allocate(8);
            List<MemorySegment> released = new ArrayList<>();
            OnnxRunInputValues values = OnnxRunInputValues.allocate(1, 1);

            values.addOwned(first);
            values.releaseOwned(released::add);
            values.releaseOwned(released::add);

            values.reset();
            values.addOwned(second);
            values.releaseOwned(released::add);

            assertEquals(List.of(first, second), released);
        }
    }

    @Test
    void uncheckedBorrowedAppendCopiesTrustedKvValuesInOrder() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment key = arena.allocate(8);
            MemorySegment value = arena.allocate(8);
            OnnxRunInputValues values = OnnxRunInputValues.allocate(2, 0);

            values.addBorrowedAllUnchecked(new MemorySegment[] { key, value }, 2);

            assertEquals(2, values.count());
            assertArrayEquals(new MemorySegment[] { key, value }, values.completeValues());
        }
    }

    @Test
    void rejectsIncompleteAndOverflowingInputs() {
        OnnxRunInputValues values = OnnxRunInputValues.allocate(2, 1);

        assertThrows(IllegalStateException.class, values::completeValues);

        values.addBorrowed(MemorySegment.NULL);
        values.addBorrowed(MemorySegment.NULL);

        assertThrows(IllegalStateException.class, () -> values.addBorrowed(MemorySegment.NULL));
        assertThrows(IllegalStateException.class, () -> values.addOwned(MemorySegment.NULL));
    }

    @Test
    void validatesConstructionAndBorrowedCounts() {
        assertThrows(IllegalArgumentException.class, () -> OnnxRunInputValues.allocate(-1, 0));
        assertThrows(IllegalArgumentException.class, () -> OnnxRunInputValues.allocate(1, -1));
        assertThrows(IllegalArgumentException.class, () -> OnnxRunInputValues.allocate(1, 2));

        OnnxRunInputValues values = OnnxRunInputValues.allocate(2, 0);
        assertThrows(IllegalArgumentException.class,
                () -> values.addBorrowedAll(new MemorySegment[] { MemorySegment.NULL }, 2));
        assertThrows(NullPointerException.class, () -> values.addBorrowedAll(null, 0));
    }

    @Test
    void bulkBorrowedAppendRejectsOverflowBeforeMutation() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment first = arena.allocate(8);
            MemorySegment second = arena.allocate(8);
            OnnxRunInputValues values = OnnxRunInputValues.allocate(1, 0);

            assertThrows(IllegalStateException.class,
                    () -> values.addBorrowedAll(new MemorySegment[] { first, second }, 2));

            assertEquals(0, values.count());

            values.addBorrowed(first);

            assertArrayEquals(new MemorySegment[] { first }, values.completeValues());
        }
    }

    @Test
    void bulkBorrowedAppendRejectsNullValuesWithIndex() {
        OnnxRunInputValues values = OnnxRunInputValues.allocate(1, 0);

        NullPointerException error = assertThrows(NullPointerException.class,
                () -> values.addBorrowedAll(new MemorySegment[] { null }, 1));

        assertEquals("borrowedValues[0]", error.getMessage());
        assertEquals(0, values.count());
    }
}
