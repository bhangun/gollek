package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

class OnnxRepeatedInputValueTest {

    @Test
    void lazilyCreatesOneValueAndRepeatsItAcrossInputs() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment repeated = arena.allocate(8);
            AtomicInteger created = new AtomicInteger();
            List<MemorySegment> released = new ArrayList<>();
            OnnxRepeatedInputValue value = OnnxRepeatedInputValue.create(
                    3,
                    () -> {
                        created.incrementAndGet();
                        return repeated;
                    },
                    released::add);
            MemorySegment[] firstTarget = new MemorySegment[5];
            MemorySegment[] secondTarget = new MemorySegment[3];

            int firstEnd = value.appendTo(firstTarget, 1);
            int secondEnd = value.appendTo(secondTarget, 0);

            assertEquals(4, firstEnd);
            assertEquals(3, secondEnd);
            assertEquals(1, created.get());
            assertSame(repeated, firstTarget[1]);
            assertSame(repeated, firstTarget[2]);
            assertSame(repeated, firstTarget[3]);
            assertNull(firstTarget[0]);
            assertNull(firstTarget[4]);
            assertSame(repeated, secondTarget[0]);

            value.release();
            value.release();

            assertEquals(List.of(repeated), released);
        }
    }

    @Test
    void zeroRepeatCountDoesNotCreateValue() {
        AtomicInteger created = new AtomicInteger();
        AtomicInteger released = new AtomicInteger();
        OnnxRepeatedInputValue value = OnnxRepeatedInputValue.create(
                0,
                () -> {
                    created.incrementAndGet();
                    return MemorySegment.NULL;
                },
                ignored -> released.incrementAndGet());

        assertEquals(2, value.appendTo(new MemorySegment[2], 2));
        value.release();

        assertEquals(0, created.get());
        assertEquals(0, released.get());
    }

    @Test
    void rejectsInvalidCountsAndTargetWindows() {
        assertThrows(IllegalArgumentException.class,
                () -> OnnxRepeatedInputValue.create(-1, () -> MemorySegment.NULL, ignored -> {
                }));

        OnnxRepeatedInputValue value = OnnxRepeatedInputValue.create(
                2,
                () -> MemorySegment.NULL,
                ignored -> {
                });

        assertThrows(IllegalArgumentException.class, () -> value.appendTo(new MemorySegment[1], 0));
        assertThrows(IllegalArgumentException.class, () -> value.appendTo(new MemorySegment[3], -1));
        assertThrows(NullPointerException.class, () -> value.appendTo(null, 0));
    }
}
