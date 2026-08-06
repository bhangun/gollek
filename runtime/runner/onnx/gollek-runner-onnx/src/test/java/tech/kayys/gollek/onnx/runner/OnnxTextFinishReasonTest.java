package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.inference.InferenceResponse;

class OnnxTextFinishReasonTest {

    @Test
    void mapsStopLengthAndCancelledToWireAndResponseReasons() {
        assertEquals("stop", OnnxTextFinishReason.STOP.wireValue());
        assertEquals(InferenceResponse.FinishReason.STOP, OnnxTextFinishReason.STOP.responseReason());

        assertEquals("length", OnnxTextFinishReason.LENGTH.wireValue());
        assertEquals(InferenceResponse.FinishReason.LENGTH, OnnxTextFinishReason.LENGTH.responseReason());

        assertEquals("cancelled", OnnxTextFinishReason.CANCELLED.wireValue());
        assertEquals(InferenceResponse.FinishReason.ERROR, OnnxTextFinishReason.CANCELLED.responseReason());
    }

    @Test
    void generationResultRequiresFinishReason() {
        assertThrows(NullPointerException.class,
                () -> new OnnxTextGenerationResult("req", "", 0, 0, 0L, null, false, null));
    }
}
