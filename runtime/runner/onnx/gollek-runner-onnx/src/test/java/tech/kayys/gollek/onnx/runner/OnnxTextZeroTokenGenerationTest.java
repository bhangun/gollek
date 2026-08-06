package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;

class OnnxTextZeroTokenGenerationTest {

    @Test
    void finishesWithoutGeneratingOutputTokens() {
        InferenceRequest request = InferenceRequest.builder()
                .model("onnx-test")
                .message(Message.user("hello"))
                .build();
        OnnxTextGenerationSetup setup = OnnxTextGenerationSetup.prepare(
                request,
                (tokenId, tokenIndex) -> {
                    throw new AssertionError("zero-token generation must not emit tokens");
                },
                null,
                OnnxInferenceProfile.start(request),
                ignored -> 0,
                ignored -> new int[] { 10, 20 },
                () -> null);

        OnnxTextGenerationResult result = OnnxTextZeroTokenGeneration.finish(setup, System.currentTimeMillis());

        assertEquals(setup.requestId(), result.requestId());
        assertEquals("", result.content());
        assertEquals(2, result.inputTokens());
        assertEquals(0, result.outputTokens());
        assertTrue(result.durationMs() >= 0L);
        assertFalse(result.fallback());
        assertEquals(OnnxTextFinishReason.LENGTH, result.finishReason());
    }

    @Test
    void validatesSetup() {
        assertThrows(NullPointerException.class, () -> OnnxTextZeroTokenGeneration.finish(null, 0L));
    }
}
