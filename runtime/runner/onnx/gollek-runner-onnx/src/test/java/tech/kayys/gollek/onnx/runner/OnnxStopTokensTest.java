package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class OnnxStopTokensTest {

    @Test
    void containsTokensWithoutBoxedSetLookup() {
        OnnxStopTokens stops = OnnxStopTokens.of(2, 42, 2);

        assertTrue(stops.contains(2));
        assertTrue(stops.contains(42));
        assertFalse(stops.contains(7));
        assertEquals(2, stops.size());
        assertArrayEquals(new int[] { 2, 42 }, stops.toArray());
    }

    @Test
    void builderMergesBaseAndAdditionalTokens() {
        OnnxStopTokens stops = OnnxStopTokens.builder(OnnxStopTokens.of(2))
                .add(42)
                .addAll(new int[] { 42, 99 })
                .build();

        assertTrue(stops.contains(2));
        assertTrue(stops.contains(42));
        assertTrue(stops.contains(99));
        assertEquals(3, stops.size());
    }
}
