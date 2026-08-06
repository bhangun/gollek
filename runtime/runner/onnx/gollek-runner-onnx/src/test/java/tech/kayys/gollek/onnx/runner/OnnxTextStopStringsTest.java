package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class OnnxTextStopStringsTest {

    @Test
    void parsesCommonStopParameterShapesInStableOrder() {
        OnnxTextStopStrings stops = OnnxTextStopStrings.fromParameters(Map.of(
                "stop", "END",
                "stop_strings", List.of("###", "END"),
                "stop_sequences", new String[] { "</s>", "" }));

        assertEquals(List.of("END", "###", "</s>"), stops.values());
        assertEquals(4, stops.maxLength());
    }

    @Test
    void matchesOnlyTextSuffixes() {
        OnnxTextStopStrings stops = OnnxTextStopStrings.fromParameters(Map.of(
                "stop", List.of("END", "###")));

        assertTrue(stops.matches("hello END"));
        assertTrue(stops.matches("hello###"));
        assertFalse(stops.matches("END then more"));
        assertFalse(stops.matches(""));
        assertFalse(stops.matches(null));
    }

    @Test
    void emptyParametersProduceNoStopStrings() {
        OnnxTextStopStrings stops = OnnxTextStopStrings.fromParameters(Map.of());

        assertTrue(stops.isEmpty());
        assertEquals(0, stops.maxLength());
        assertFalse(stops.matches("anything"));
    }

    @Test
    void shortTextDoesNotScanStopSuffixes() {
        OnnxTextStopStrings stops = OnnxTextStopStrings.fromParameters(Map.of(
                "stop", List.of("END", "DONE")));

        assertFalse(stops.matches("NO"));
        assertTrue(stops.matches("DONE"));
    }
}
