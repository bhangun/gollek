package tech.kayys.golek.provider.gemini;

import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.Multi;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.wayang.tenant.TenantContext;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

class GeminiProviderTest {

    @InjectMocks
    private GeminiProvider geminiProvider;

    @Mock
    private GeminiClient geminiClient;

    @Mock
    private GeminiConfig geminiConfig;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void id() {
        assertEquals("gemini", geminiProvider.id());
    }

    @Test
    void name() {
        assertEquals("Google Gemini", geminiProvider.name());
    }

    @Test
    void version() {
        assertEquals("1.0.0", geminiProvider.version());
    }

    @Test
    void supports() {
        assertTrue(geminiProvider.supports("gemini-1.5-pro", null));
        assertTrue(geminiProvider.supports("gemini-2.0-flash-exp", null));
        assertFalse(geminiProvider.supports("some-other-model", null));
    }

    @Test
    void doInfer() {
        ProviderRequest request = new ProviderRequest();
        request.setModel("gemini-1.5-pro");
        request.setMessages(Collections.singletonList(new tech.kayys.golek.api.model.Message("user", "Hello")));

        GeminiResponse geminiResponse = new GeminiResponse();
        GeminiCandidate candidate = new GeminiCandidate();
        GeminiContent content = new GeminiContent();
        content.setParts(Collections.singletonList(new GeminiPart("Hello, world!")));
        candidate.setContent(content);
        geminiResponse.setCandidates(Collections.singletonList(candidate));
        GeminiUsageMetadata usageMetadata = new GeminiUsageMetadata();
        usageMetadata.setPromptTokenCount(1);
        usageMetadata.setCandidatesTokenCount(2);
        usageMetadata.setTotalTokenCount(3);
        geminiResponse.setUsageMetadata(usageMetadata);

        when(geminiClient.generateContent(any(), any(), any())).thenReturn(Uni.createFrom().item(geminiResponse));

        InferenceResponse response = geminiProvider.infer(request, null).await().indefinitely();

        assertNotNull(response);
        assertEquals("Hello, world!", response.getContent());
        assertEquals(3, response.getMetadata().get("total_tokens"));
    }

    @Test
    void stream() {
        ProviderRequest request = new ProviderRequest();
        request.setModel("gemini-1.5-pro");
        request.setMessages(Collections.singletonList(new tech.kayys.golek.api.model.Message("user", "Hello")));

        GeminiResponse chunk1 = new GeminiResponse();
        GeminiCandidate candidate1 = new GeminiCandidate();
        GeminiContent content1 = new GeminiContent();
        content1.setParts(List.of(new GeminiPart("Hello, ")));
        candidate1.setContent(content1);
        chunk1.setCandidates(List.of(candidate1));

        GeminiResponse chunk2 = new GeminiResponse();
        GeminiCandidate candidate2 = new GeminiCandidate();
        GeminiContent content2 = new GeminiContent();
        content2.setParts(List.of(new GeminiPart("world!")));
        candidate2.setContent(content2);
        candidate2.setFinishReason("STOP");
        chunk2.setCandidates(List.of(candidate2));

        when(geminiClient.streamGenerateContent(any(), any(), any()))
                .thenReturn(Multi.createFrom().items(chunk1, chunk2));

        List<StreamChunk> chunks = geminiProvider.stream(request, null).collect().asList().await().indefinitely();

        assertNotNull(chunks);
        assertEquals(2, chunks.size());
        assertEquals("Hello, ", chunks.get(0).getDelta());
        assertEquals("world!", chunks.get(1).getDelta());
        assertTrue(chunks.get(1).isFinal());
    }

    @Test
    void healthCheck() {
        when(geminiClient.listModels(any())).thenReturn(Uni.createFrom().item(Collections.emptyMap()));
        assertTrue(geminiProvider.healthCheck().await().indefinitely().isHealthy());
    }
}