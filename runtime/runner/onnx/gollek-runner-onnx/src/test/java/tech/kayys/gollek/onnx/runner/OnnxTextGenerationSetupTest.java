package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;

class OnnxTextGenerationSetupTest {

    @Test
    void preparesRequestDefaultsAndSkipsStopDecoderWhenNoStopsExist() {
        AtomicInteger stopDecoderCreations = new AtomicInteger();
        InferenceRequest request = request(Map.of());

        OnnxTextGenerationSetup setup = OnnxTextGenerationSetup.prepare(
                request,
                null,
                null,
                OnnxInferenceProfile.start(request),
                ignored -> 7,
                ignored -> new int[] { 1, 2 },
                () -> {
                    stopDecoderCreations.incrementAndGet();
                    return OnnxStreamingTokenDecoder.create(null, tokenId -> "<" + tokenId + ">");
                });

        assertEquals(request.getRequestId(), setup.requestId());
        assertEquals(7, setup.maxTokens());
        assertEquals(2, setup.promptLength());
        assertFalse(setup.cancellation().getAsBoolean());
        assertNull(setup.stopStringDecoder());
        assertEquals(0, stopDecoderCreations.get());
    }

    @Test
    void createsStopDecoderOnlyWhenStopStringsExist() {
        AtomicInteger stopDecoderCreations = new AtomicInteger();
        InferenceRequest request = request(Map.of("stop", "done"));

        OnnxTextGenerationSetup setup = OnnxTextGenerationSetup.prepare(
                request,
                OnnxGeneratedTokenObserver.NOOP,
                () -> true,
                OnnxInferenceProfile.start(request),
                ignored -> 1,
                ignored -> new int[] { 9 },
                () -> {
                    stopDecoderCreations.incrementAndGet();
                    return OnnxStreamingTokenDecoder.create(null, tokenId -> "done");
                });

        assertTrue(setup.cancellation().getAsBoolean());
        assertEquals(1, stopDecoderCreations.get());
        assertNotNull(setup.stopStringDecoder());
        assertTrue(setup.stopStrings().matches("done"));
    }

    @Test
    void validatesPreparedValues() {
        InferenceRequest request = request(Map.of());
        OnnxInferenceProfile profile = OnnxInferenceProfile.start(request);

        assertThrows(IllegalArgumentException.class,
                () -> OnnxTextGenerationSetup.prepare(
                        request,
                        null,
                        null,
                        profile,
                        ignored -> -1,
                        ignored -> new int[] { 1 },
                        () -> null));
        assertThrows(IllegalArgumentException.class,
                () -> OnnxTextGenerationSetup.prepare(
                        request,
                        null,
                        null,
                        profile,
                        ignored -> 1,
                        ignored -> new int[0],
                        () -> null));
        assertThrows(NullPointerException.class,
                () -> OnnxTextGenerationSetup.prepare(
                        null,
                        null,
                        null,
                        profile,
                        ignored -> 1,
                        ignored -> new int[] { 1 },
                        () -> null));
    }

    @Test
    void acceptsZeroMaxTokensAsNoGenerationBudget() {
        InferenceRequest request = request(Map.of());

        OnnxTextGenerationSetup setup = OnnxTextGenerationSetup.prepare(
                request,
                null,
                null,
                OnnxInferenceProfile.start(request),
                ignored -> 0,
                ignored -> new int[] { 1 },
                () -> null);

        assertEquals(0, setup.maxTokens());
    }

    @Test
    void carriesOptionalFinalContentSupplier() {
        InferenceRequest request = request(Map.of());

        OnnxTextGenerationSetup setup = OnnxTextGenerationSetup.prepare(
                request,
                null,
                null,
                OnnxInferenceProfile.start(request),
                ignored -> 1,
                ignored -> new int[] { 1 },
                () -> null,
                () -> "already decoded");

        assertEquals("already decoded", setup.finalContentOrNull());
    }

    private static InferenceRequest request(Map<String, Object> parameters) {
        InferenceRequest.Builder builder = InferenceRequest.builder()
                .model("onnx-test")
                .message(Message.user("hello"));
        parameters.forEach(builder::parameter);
        return builder.build();
    }
}
