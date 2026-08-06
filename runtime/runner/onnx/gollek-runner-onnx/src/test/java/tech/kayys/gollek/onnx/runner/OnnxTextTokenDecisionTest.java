package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class OnnxTextTokenDecisionTest {

    @Test
    void continueDecisionIsReusable() {
        OnnxTextTokenDecision first = OnnxTextTokenDecision.continueGeneration();
        OnnxTextTokenDecision second = OnnxTextTokenDecision.continueGeneration();

        assertSame(first, second);
        assertEquals(OnnxTextFinishReason.LENGTH, first.finishReason());
    }

    @Test
    void stopDecisionsAreReusableByFinishReason() {
        for (OnnxTextFinishReason reason : OnnxTextFinishReason.values()) {
            OnnxTextTokenDecision first = OnnxTextTokenDecision.stop(reason);
            OnnxTextTokenDecision second = OnnxTextTokenDecision.stop(reason);

            assertSame(first, second);
            assertTrue(first.finished());
            assertEquals(reason, first.finishReason());
        }
    }

    @Test
    void stopDecisionRequiresFinishReason() {
        assertThrows(NullPointerException.class, () -> OnnxTextTokenDecision.stop(null));
    }
}
