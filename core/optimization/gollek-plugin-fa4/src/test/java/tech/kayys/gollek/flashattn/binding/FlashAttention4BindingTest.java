package tech.kayys.gollek.flashattn.binding;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.*;

class FlashAttention4BindingTest {

    @Test
    void cpuFallbackLaunches() {
        FlashAttention4Binding.initializeFallback();
        FlashAttention4Binding binding = FlashAttention4Binding.getInstance();
        assertFalse(binding.isNativeAvailable());

        int batch = 1, seq = 2, heads = 1, headsK = 1, dim = 2;
        int total = batch * seq * heads * dim;
        int kvTotal = batch * seq * headsK * dim;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate((long) total * 4L, 4);
            MemorySegment q = arena.allocate((long) total * 4L, 4);
            MemorySegment k = arena.allocate((long) kvTotal * 4L, 4);
            MemorySegment v = arena.allocate((long) kvTotal * 4L, 4);

            for (int i = 0; i < total; i++) q.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
            for (int i = 0; i < kvTotal; i++) {
                k.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
                v.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
            }

            int rc = binding.flashAttention4Launch(
                    out, q, k, v, batch, seq, heads, headsK, dim, 1.0f, false, false);
            assertEquals(0, rc);
            assertTrue(Float.isFinite(out.getAtIndex(ValueLayout.JAVA_FLOAT, 0)));
        }
    }
}
