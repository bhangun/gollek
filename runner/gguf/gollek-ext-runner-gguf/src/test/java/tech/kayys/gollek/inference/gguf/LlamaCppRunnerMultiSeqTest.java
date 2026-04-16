package tech.kayys.gollek.inference.gguf;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ArtifactLocation;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;

class LlamaCppRunnerMultiSeqTest {

    @Test
    void multiSequenceBatchingProducesIndependentResponses() throws Exception {
        GGUFProviderConfig config = Mockito.mock(GGUFProviderConfig.class);
        when(config.maxConcurrentRequests()).thenReturn(8);
        when(config.maxContextTokens()).thenReturn(128);
        when(config.coalesceEnabled()).thenReturn(true);
        when(config.coalesceWindowMs()).thenReturn(1);
        when(config.coalesceMaxBatch()).thenReturn(4);
        when(config.coalesceMaxQueue()).thenReturn(16);
        when(config.coalesceSeqMax()).thenReturn(2);

        LlamaCppBinding binding = Mockito.mock(LlamaCppBinding.class);
        GGUFChatTemplateService templateService = Mockito.mock(GGUFChatTemplateService.class);

        float[] logitsArray = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        Arena arena = Arena.ofAuto();
        MemorySegment logits = arena.allocate(ValueLayout.JAVA_FLOAT, logitsArray.length);
        MemorySegment heap = MemorySegment.ofArray(logitsArray);
        MemorySegment.copy(heap, 0, logits, 0, logitsArray.length * ValueLayout.JAVA_FLOAT.byteSize());

        when(binding.tokenize(any(), anyString(), anyBoolean(), anyBoolean()))
                .thenAnswer(invocation -> {
                    String prompt = invocation.getArgument(1, String.class);
                    if (prompt.contains("two")) {
                        return new int[] { 1 };
                    }
                    return new int[] { 1, 2 };
                });
        when(binding.batchInit(anyInt(), anyInt(), anyInt())).thenReturn(MemorySegment.NULL);
        when(binding.decode(any(), any())).thenReturn(0);
        when(binding.getLogitsIth(any(), anyInt())).thenReturn(logits);
        when(binding.tokenToPiece(any(), anyInt())).thenReturn("x");
        when(binding.isEndOfGeneration(any(), anyInt())).thenReturn(false);

        LlamaCppRunner runner = new LlamaCppRunner(binding, config, templateService);
        setField(runner, "context", MemorySegment.NULL);
        setField(runner, "model", MemorySegment.NULL);
        setField(runner, "vocabSize", 4);
        setField(runner, "eosToken", -1);
        setField(runner, "manifest", createManifest());

        InferenceRequest reqOne = InferenceRequest.builder()
                .model("test-model")
                .message(tech.kayys.gollek.spi.Message.user("hello one"))
                .parameter("prompt", "hello one")
                .parameter("temperature", 0.0f)
                .parameter("max_tokens", 1)
                .build();
        InferenceRequest reqTwo = InferenceRequest.builder()
                .model("test-model")
                .message(tech.kayys.gollek.spi.Message.user("hello two"))
                .parameter("prompt", "hello two")
                .parameter("temperature", 0.0f)
                .parameter("max_tokens", 1)
                .build();

        List<CompletableFuture<tech.kayys.gollek.spi.inference.InferenceResponse>> futures =
                runner.runMultiSequenceForTests(List.of(reqOne, reqTwo));

        tech.kayys.gollek.spi.inference.InferenceResponse resOne = futures.get(0).join();
        tech.kayys.gollek.spi.inference.InferenceResponse resTwo = futures.get(1).join();

        assertThat(resOne.getContent()).isEqualTo("x");
        assertThat(resTwo.getContent()).isEqualTo("x");
        assertThat(resOne.getOutputTokens()).isEqualTo(1);
        assertThat(resTwo.getOutputTokens()).isEqualTo(1);
    }

    @Test
    void multiSequenceClearsKvCacheAndTruncatesPrompts() throws Exception {
        GGUFProviderConfig config = Mockito.mock(GGUFProviderConfig.class);
        when(config.maxConcurrentRequests()).thenReturn(8);
        when(config.maxContextTokens()).thenReturn(2);
        when(config.coalesceEnabled()).thenReturn(true);
        when(config.coalesceWindowMs()).thenReturn(1);
        when(config.coalesceMaxBatch()).thenReturn(4);
        when(config.coalesceMaxQueue()).thenReturn(16);
        when(config.coalesceSeqMax()).thenReturn(2);

        LlamaCppBinding binding = Mockito.mock(LlamaCppBinding.class);
        GGUFChatTemplateService templateService = Mockito.mock(GGUFChatTemplateService.class);

        float[] logitsArray = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        Arena arena = Arena.ofAuto();
        MemorySegment logits = arena.allocate(ValueLayout.JAVA_FLOAT, logitsArray.length);
        MemorySegment heap = MemorySegment.ofArray(logitsArray);
        MemorySegment.copy(heap, 0, logits, 0, logitsArray.length * ValueLayout.JAVA_FLOAT.byteSize());

        when(binding.tokenize(any(), anyString(), anyBoolean(), anyBoolean()))
                .thenReturn(new int[] { 1, 2, 3 });
        when(binding.batchInit(anyInt(), anyInt(), anyInt())).thenReturn(MemorySegment.NULL);
        when(binding.decode(any(), any())).thenReturn(0);
        when(binding.getLogitsIth(any(), anyInt())).thenReturn(logits);
        when(binding.tokenToPiece(any(), anyInt())).thenReturn("x");
        when(binding.isEndOfGeneration(any(), anyInt())).thenReturn(false);

        LlamaCppRunner runner = new LlamaCppRunner(binding, config, templateService);
        setField(runner, "context", MemorySegment.NULL);
        setField(runner, "model", MemorySegment.NULL);
        setField(runner, "vocabSize", 4);
        setField(runner, "eosToken", -1);
        setField(runner, "manifest", createManifest());

        InferenceRequest reqOne = InferenceRequest.builder()
                .model("test-model")
                .message(tech.kayys.gollek.spi.Message.user("long prompt one"))
                .parameter("prompt", "long prompt one")
                .parameter("temperature", 0.0f)
                .parameter("max_tokens", 1)
                .build();
        InferenceRequest reqTwo = InferenceRequest.builder()
                .model("test-model")
                .message(tech.kayys.gollek.spi.Message.user("long prompt two"))
                .parameter("prompt", "long prompt two")
                .parameter("temperature", 0.0f)
                .parameter("max_tokens", 1)
                .build();

        List<CompletableFuture<tech.kayys.gollek.spi.inference.InferenceResponse>> futures =
                runner.runMultiSequenceForTests(List.of(reqOne, reqTwo));

        assertThat(futures.get(0).join().getContent()).isEqualTo("");
        assertThat(futures.get(1).join().getContent()).isEqualTo("");
        Mockito.verify(binding, Mockito.atLeastOnce()).kvCacheClear(any());
    }

    @Test
    void multiSequenceClampsMaxTokensToContextWindow() throws Exception {
        GGUFProviderConfig config = Mockito.mock(GGUFProviderConfig.class);
        when(config.maxConcurrentRequests()).thenReturn(8);
        when(config.maxContextTokens()).thenReturn(1);
        when(config.coalesceEnabled()).thenReturn(true);
        when(config.coalesceWindowMs()).thenReturn(1);
        when(config.coalesceMaxBatch()).thenReturn(4);
        when(config.coalesceMaxQueue()).thenReturn(16);
        when(config.coalesceSeqMax()).thenReturn(2);

        LlamaCppBinding binding = Mockito.mock(LlamaCppBinding.class);
        GGUFChatTemplateService templateService = Mockito.mock(GGUFChatTemplateService.class);

        float[] logitsArray = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        Arena arena = Arena.ofAuto();
        MemorySegment logits = arena.allocate(ValueLayout.JAVA_FLOAT, logitsArray.length);
        MemorySegment heap = MemorySegment.ofArray(logitsArray);
        MemorySegment.copy(heap, 0, logits, 0, logitsArray.length * ValueLayout.JAVA_FLOAT.byteSize());

        when(binding.tokenize(any(), anyString(), anyBoolean(), anyBoolean()))
                .thenReturn(new int[] { 1 });
        when(binding.batchInit(anyInt(), anyInt(), anyInt())).thenReturn(MemorySegment.NULL);
        when(binding.decode(any(), any())).thenReturn(0);
        when(binding.getLogitsIth(any(), anyInt())).thenReturn(logits);
        when(binding.tokenToPiece(any(), anyInt())).thenReturn("x");
        when(binding.isEndOfGeneration(any(), anyInt())).thenReturn(false);

        LlamaCppRunner runner = new LlamaCppRunner(binding, config, templateService);
        setField(runner, "context", MemorySegment.NULL);
        setField(runner, "model", MemorySegment.NULL);
        setField(runner, "vocabSize", 4);
        setField(runner, "eosToken", -1);
        setField(runner, "manifest", createManifest());

        InferenceRequest reqOne = InferenceRequest.builder()
                .model("test-model")
                .message(tech.kayys.gollek.spi.Message.user("short prompt"))
                .parameter("prompt", "short prompt")
                .parameter("temperature", 0.0f)
                .parameter("max_tokens", 4)
                .build();
        InferenceRequest reqTwo = InferenceRequest.builder()
                .model("test-model")
                .message(tech.kayys.gollek.spi.Message.user("short prompt 2"))
                .parameter("prompt", "short prompt 2")
                .parameter("temperature", 0.0f)
                .parameter("max_tokens", 4)
                .build();

        List<CompletableFuture<tech.kayys.gollek.spi.inference.InferenceResponse>> futures =
                runner.runMultiSequenceForTests(List.of(reqOne, reqTwo));

        assertThat(futures.get(0).join().getOutputTokens()).isEqualTo(0);
        assertThat(futures.get(1).join().getOutputTokens()).isEqualTo(0);
        assertThat(futures.get(0).join().getContent()).isEqualTo("");
        assertThat(futures.get(1).join().getContent()).isEqualTo("");
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static ModelManifest createManifest() {
        ArtifactLocation location = new ArtifactLocation("test.gguf", null, null, null);
        return ModelManifest.builder()
                .modelId("test-model")
                .name("test-model")
                .version("test")
                .path(location.uri())
                .apiKey(tech.kayys.gollek.spi.auth.ApiKeyConstants.COMMUNITY_API_KEY)
                .requestId("tenant1")
                .artifacts(Map.of(ModelFormat.GGUF, location))
                .supportedDevices(List.of())
                .resourceRequirements(null)
                .metadata(Map.of())
                .createdAt(java.time.Instant.now())
                .updatedAt(java.time.Instant.now())
                .build();
    }
}
