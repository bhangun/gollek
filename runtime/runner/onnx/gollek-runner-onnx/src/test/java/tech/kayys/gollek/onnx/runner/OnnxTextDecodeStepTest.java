package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class OnnxTextDecodeStepTest {

    @Test
    void plansKvPrefillFromPromptLength() {
        OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(true, false, 5, 5, 0);

        assertTrue(step.prefill());
        assertEquals(5L, step.sequenceLength());
        assertEquals(5L, step.attentionLength());
        assertEquals(0L, step.positionStart());
    }

    @Test
    void plansKvDecodeFromConsumedTokens() {
        OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(true, true, 5, 6, 5);

        assertFalse(step.prefill());
        assertEquals(1L, step.sequenceLength());
        assertEquals(6L, step.attentionLength());
        assertEquals(5L, step.positionStart());
    }

    @Test
    void plansStatelessDecodeFromTokenHistory() {
        OnnxTextDecodeStep step = OnnxTextDecodeStep.plan(false, false, 5, 6, 99);

        assertFalse(step.prefill());
        assertEquals(6L, step.sequenceLength());
        assertEquals(6L, step.attentionLength());
        assertEquals(0L, step.positionStart());
    }

    @Test
    void updateReusesExistingPlanInstance() {
        OnnxTextDecodeStep step = OnnxTextDecodeStep.reusable();

        assertSame(step, step.update(true, false, 5, 5, 0));
        assertTrue(step.prefill());
        assertEquals(5L, step.sequenceLength());

        assertSame(step, step.update(true, true, 5, 6, 5));
        assertFalse(step.prefill());
        assertEquals(1L, step.sequenceLength());
        assertEquals(6L, step.attentionLength());
        assertEquals(5L, step.positionStart());
    }

    @Test
    void validatesTokenCounts() {
        assertThrows(IllegalArgumentException.class, () -> OnnxTextDecodeStep.plan(true, false, 0, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> OnnxTextDecodeStep.plan(true, false, 1, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> OnnxTextDecodeStep.plan(true, false, 1, 1, -1));
    }
}
