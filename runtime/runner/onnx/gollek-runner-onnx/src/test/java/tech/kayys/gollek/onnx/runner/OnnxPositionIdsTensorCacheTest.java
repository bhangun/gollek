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

class OnnxPositionIdsTensorCacheTest {

    @Test
    void evictsLeastRecentlyUsedPositionRangeWithoutRecordKeys() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxPositionIdsTensorCache cache = new OnnxPositionIdsTensorCache(2);
            List<MemorySegment> released = new ArrayList<>();
            MemorySegment zeroToOne = arena.allocate(Long.BYTES);
            MemorySegment oneToThree = arena.allocate(Long.BYTES * 2L);
            MemorySegment threeToFour = arena.allocate(Long.BYTES);

            assertNull(cache.get(0, 1));
            assertTrue(cache.retain(0, 1, zeroToOne, released::add));
            assertNull(cache.get(1, 2));
            assertTrue(cache.retain(1, 2, oneToThree, released::add));
            assertSame(zeroToOne, cache.get(0, 1));
            assertNull(cache.get(3, 1));
            assertTrue(cache.retain(3, 1, threeToFour, released::add));

            assertEquals(List.of(oneToThree), released);
            assertSame(zeroToOne, cache.get(0, 1));
            assertSame(threeToFour, cache.get(3, 1));
            assertEquals(3, cache.hits());
            assertEquals(3, cache.misses());
            assertEquals(1, cache.evictions());
        }
    }

    @Test
    void refreshingSameSegmentDoesNotReleaseKeptTensor() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxPositionIdsTensorCache cache = new OnnxPositionIdsTensorCache(2);
            List<MemorySegment> released = new ArrayList<>();
            MemorySegment tensor = arena.allocate(Long.BYTES);

            assertTrue(cache.retain(1, 1, tensor, released::add));
            assertTrue(cache.retain(1, 1, tensor, released::add));

            assertEquals(List.of(), released);
            assertSame(tensor, cache.get(1, 1));
        }
    }

    @Test
    void zeroCapacityDoesNotRetainCreatedTensor() {
        try (Arena arena = Arena.ofConfined()) {
            OnnxPositionIdsTensorCache cache = new OnnxPositionIdsTensorCache(0);
            List<MemorySegment> released = new ArrayList<>();
            MemorySegment tensor = arena.allocate(Long.BYTES);

            assertNull(cache.get(1, 1));
            assertFalse(cache.retain(1, 1, tensor, released::add));
            cache.releaseAll(released::add);

            assertEquals(List.of(), released);
            assertEquals(0, cache.hits());
            assertEquals(0, cache.misses());
            assertEquals(0, cache.evictions());
        }
    }
}
