package tech.kayys.golek.plugin.observability;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link InferenceTrace}.
 */
class InferenceTraceTest {

    @Test
    void totalTokens_sumsInputAndOutput() {
        var trace = new InferenceTrace("req-1", Instant.now());
        trace.setInputTokens(100);
        trace.setOutputTokens(50);

        assertEquals(150, trace.getTotalTokens());
    }

    @Test
    void hasError_reflectsErrorState() {
        var trace = new InferenceTrace("req-1", Instant.now());
        assertFalse(trace.hasError());

        trace.setError("Something went wrong");
        assertTrue(trace.hasError());
    }

    @Test
    void toolUsage_tracksMultipleTools() {
        var trace = new InferenceTrace("req-1", Instant.now());
        trace.addToolUsage("search");
        trace.addToolUsage("calculator");

        assertEquals(2, trace.getToolsUsed().size());
        assertTrue(trace.getToolsUsed().contains("search"));
        assertTrue(trace.getToolsUsed().contains("calculator"));
    }

    @Test
    void phaseLatency_tracksMultiplePhases() {
        var trace = new InferenceTrace("req-1", Instant.now());
        trace.addPhaseLatency("VALIDATE", Duration.ofMillis(50));
        trace.addPhaseLatency("PRE_PROCESSING", Duration.ofMillis(100));

        assertEquals(2, trace.getPhaseLatencies().size());
    }

    @Test
    void toString_containsRequestId() {
        var trace = new InferenceTrace("req-42", Instant.now());
        assertTrue(trace.toString().contains("req-42"));
    }
}
