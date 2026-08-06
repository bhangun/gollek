package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class OnnxLengthTensorCacheTest {

    @Test
    void evictsLeastRecentlyUsedLengthWithoutBoxedKeys() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxLengthTensorCache cache = new OnnxLengthTensorCache(2);
            List<MemorySegment> released = new ArrayList<>();
            MemorySegment one = arena.allocate(Long.BYTES);
            MemorySegment two = arena.allocate(Long.BYTES);
            MemorySegment three = arena.allocate(Long.BYTES);

            assertNull(cache.get(1));
            assertTrue(cache.retain(1, one, released::add));
            assertNull(cache.get(2));
            assertTrue(cache.retain(2, two, released::add));
            assertSame(one, cache.get(1));
            assertNull(cache.get(3));
            assertTrue(cache.retain(3, three, released::add));

            assertEquals(List.of(two), released);
            assertSame(one, cache.get(1));
            assertSame(three, cache.get(3));
            assertEquals(3, cache.hits());
            assertEquals(3, cache.misses());
            assertEquals(1, cache.evictions());
        }
    }

    @Test
    void refreshingSameSegmentDoesNotReleaseKeptTensor() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxLengthTensorCache cache = new OnnxLengthTensorCache(2);
            List<MemorySegment> released = new ArrayList<>();
            MemorySegment tensor = arena.allocate(Long.BYTES);

            assertTrue(cache.retain(1, tensor, released::add));
            assertTrue(cache.retain(1, tensor, released::add));

            assertEquals(List.of(), released);
            assertSame(tensor, cache.get(1));
        }
    }

    @Test
    void zeroCapacityDoesNotRetainCreatedTensor() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxLengthTensorCache cache = new OnnxLengthTensorCache(0);
            List<MemorySegment> released = new ArrayList<>();
            MemorySegment tensor = arena.allocate(Long.BYTES);

            assertNull(cache.get(1));
            assertFalse(cache.retain(1, tensor, released::add));
            cache.releaseAll(released::add);

            assertEquals(List.of(), released);
            assertEquals(0, cache.hits());
            assertEquals(0, cache.misses());
            assertEquals(0, cache.evictions());
        }
    }
}
