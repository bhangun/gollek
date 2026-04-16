package tech.kayys.gollek.qlora.binding;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.*;

class QLoraCpuFallbackTest {

    @Test
    void loadNf4WritesPackedAndScales() {
        int N = 1;
        int K = 64;
        int packedStride = K / 2;
        int scalesPerRow = (K + 63) / 64;
        long packedBytes = (long) N * packedStride;
        long scalesBytes = (long) N * scalesPerRow * 2L;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment src = arena.allocate((long) N * K * 4L, 4);
            for (int i = 0; i < N * K; i++) {
                src.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) i);
            }

            MemorySegment dst = arena.allocate(packedBytes + scalesBytes, 4);
            int rc = QLoraCpuFallback.loadNf4(dst, src, N, K);
            assertEquals(0, rc);

            boolean anyPackedNonZero = false;
            for (int i = 0; i < packedBytes; i++) {
                if (dst.getAtIndex(ValueLayout.JAVA_BYTE, i) != 0) {
                    anyPackedNonZero = true;
                    break;
                }
            }
            assertTrue(anyPackedNonZero, "expected some packed bytes to be non-zero");

            short scale0 = dst.getAtIndex(ValueLayout.JAVA_SHORT, packedBytes / 2);
            assertNotEquals(0, scale0);
        }
    }
}
