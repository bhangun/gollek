package tech.kayys.golek.plugin.sampling;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link SamplingConfig}.
 */
class SamplingConfigTest {

    @Test
    void defaults_returnsExpectedValues() {
        SamplingConfig config = SamplingConfig.defaults();

        assertEquals(0.7, config.temperature(), 0.01);
        assertEquals(40, config.topK());
        assertEquals(0.95, config.topP(), 0.01);
        assertEquals(1.1, config.repetitionPenalty(), 0.01);
        assertEquals(0.0, config.presencePenalty(), 0.01);
        assertEquals(2048, config.maxTokens());
        assertTrue(config.stopTokens().isEmpty());
        assertNull(config.grammarMode());
    }

    @Test
    void deterministic_returnsZeroTemperature() {
        SamplingConfig config = SamplingConfig.deterministic();

        assertEquals(0.0, config.temperature(), 0.01);
        assertEquals(1, config.topK());
    }

    @Test
    void creative_returnsHighTemperature() {
        SamplingConfig config = SamplingConfig.creative();

        assertTrue(config.temperature() > 1.0);
        assertEquals(4096, config.maxTokens());
    }

    @Test
    void jsonMode_setsGrammarMode() {
        SamplingConfig config = SamplingConfig.jsonMode();

        assertEquals("json", config.grammarMode());
        assertEquals(0.0, config.temperature(), 0.01);
    }
}
