package tech.kayys.gollek.onnx.runner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;

class OnnxTextCpuFallbackGenerationTest {

    @Test
    void producesFallbackGenerationResultAndNotifiesObserver() {
        FakeOps ops = new FakeOps();
        List<String> observed = new ArrayList<>();
        InferenceRequest request = InferenceRequest.builder()
                .model("onnx-test")
                .message(Message.user("hello"))
                .build();
        OnnxTextGenerationSetup setup = OnnxTextGenerationSetup.prepare(
                request,
                (tokenId, tokenIndex) -> {
                    observed.add(tokenId + "@" + tokenIndex);
                    return null;
                },
                null,
                OnnxInferenceProfile.start(request),
                ignored -> 5,
                ignored -> new int[] { 10, 20, 30 },
                () -> null);
        OnnxTextCpuFallbackGeneration fallback = OnnxTextCpuFallbackGeneration.createForTest(
                ops,
                4,
                logits -> {
                    assertEquals(4, logits.length);
                    return 2;
                },
                tokenId -> "token-" + tokenId);

        OnnxTextGenerationResult result = fallback.generate(setup, System.currentTimeMillis());

        assertEquals(setup.requestId(), result.requestId());
        assertEquals("token-2", result.content());
        assertEquals(3, result.inputTokens());
        assertEquals(1, result.outputTokens());
        assertTrue(result.durationMs() >= 0L);
        assertTrue(result.fallback());
        assertEquals(OnnxTextFinishReason.STOP, result.finishReason());
        assertEquals(List.of("2@0"), observed);
        assertEquals(List.of(4), ops.outputRequests);
    }

    @Test
    void reusesObservedFinalContentWhenAvailable() {
        FakeOps ops = new FakeOps();
        List<String> observed = new ArrayList<>();
        InferenceRequest request = InferenceRequest.builder()
                .model("onnx-test")
                .message(Message.user("hello"))
                .build();
        OnnxTextGenerationSetup setup = OnnxTextGenerationSetup.prepare(
                request,
                (tokenId, tokenIndex) -> {
                    observed.add(tokenId + "@" + tokenIndex);
                    return "streamed-token-" + tokenId;
                },
                null,
                OnnxInferenceProfile.start(request),
                ignored -> 5,
                ignored -> new int[] { 10 },
                () -> null,
                () -> "streamed-token-2");
        OnnxTextCpuFallbackGeneration fallback = OnnxTextCpuFallbackGeneration.createForTest(
                ops,
                4,
                logits -> 2,
                tokenId -> {
                    throw new AssertionError("fallback decoder should not be used when final content is available");
                });

        OnnxTextGenerationResult result = fallback.generate(setup, System.currentTimeMillis());

        assertEquals("streamed-token-2", result.content());
        assertEquals(List.of("2@0"), observed);
        assertEquals(List.of(4), ops.outputRequests);
    }

    @Test
    void validatesConstructionAndGenerateInput() {
        FakeOps ops = new FakeOps();

        assertThrows(IllegalArgumentException.class,
                () -> OnnxTextCpuFallbackGeneration.createForTest(ops, 0, logits -> 0, tokenId -> ""));
        assertThrows(NullPointerException.class,
                () -> OnnxTextCpuFallbackGeneration.createForTest(null, 1, logits -> 0, tokenId -> ""));
        assertThrows(NullPointerException.class,
                () -> OnnxTextCpuFallbackGeneration.createForTest(ops, 1, null, tokenId -> ""));
        assertThrows(NullPointerException.class,
                () -> OnnxTextCpuFallbackGeneration.createForTest(ops, 1, logits -> 0, null));

        OnnxTextCpuFallbackGeneration fallback = OnnxTextCpuFallbackGeneration.createForTest(
                ops,
                1,
                logits -> 0,
                tokenId -> "");
        assertThrows(NullPointerException.class, () -> fallback.generate(null, 0L));
    }

    private static final class FakeOps implements OnnxTextCpuFallbackGeneration.Ops {
        private final List<Integer> outputRequests = new ArrayList<>();

        @Override
        public float[] run(int outputFloats) {
            outputRequests.add(outputFloats);
            return new float[outputFloats];
        }
    }
}
