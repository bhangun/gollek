package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class OnnxTextGenerationProgressTest {

    @Test
    void startsWithLengthFinishAndNoConsumedTokens() {
        OnnxTextGenerationProgress progress = new OnnxTextGenerationProgress();

        assertEquals(0L, progress.consumedTokens());
        assertEquals(OnnxTextFinishReason.LENGTH, progress.finishReason());
    }

    @Test
    void advancesConsumedTokens() {
        OnnxTextGenerationProgress progress = new OnnxTextGenerationProgress();

        progress.advance(5);
        progress.advance(1);

        assertEquals(6L, progress.consumedTokens());
    }

    @Test
    void rejectsInvalidSequenceLengths() {
        OnnxTextGenerationProgress progress = new OnnxTextGenerationProgress();

        assertThrows(IllegalArgumentException.class, () -> progress.advance(0));
        assertThrows(IllegalArgumentException.class, () -> progress.advance(-1));
    }

    @Test
    void cancelSetsCancelledFinishReason() {
        OnnxTextGenerationProgress progress = new OnnxTextGenerationProgress();

        progress.cancel();

        assertEquals(OnnxTextFinishReason.CANCELLED, progress.finishReason());
    }

    @Test
    void applyContinueKeepsLengthFinishReason() {
        OnnxTextGenerationProgress progress = new OnnxTextGenerationProgress();

        assertFalse(progress.apply(OnnxTextTokenDecision.continueGeneration()));
        assertEquals(OnnxTextFinishReason.LENGTH, progress.finishReason());
    }

    @Test
    void applyStopStoresTerminalFinishReason() {
        OnnxTextGenerationProgress progress = new OnnxTextGenerationProgress();

        assertTrue(progress.apply(OnnxTextTokenDecision.stop(OnnxTextFinishReason.STOP)));

        assertEquals(OnnxTextFinishReason.STOP, progress.finishReason());
    }

    @Test
    void validatesTokenDecision() {
        OnnxTextGenerationProgress progress = new OnnxTextGenerationProgress();

        assertThrows(NullPointerException.class, () -> progress.apply(null));
    }
}
