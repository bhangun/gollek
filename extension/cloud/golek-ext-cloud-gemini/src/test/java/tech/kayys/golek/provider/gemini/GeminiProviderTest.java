package tech.kayys.golek.provider.gemini;

import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.Multi;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.ProviderRequest;
import tech.kayys.golek.spi.stream.StreamChunk;
import java.util.Collections;
import java.util.List;
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
    void supports() {
        assertTrue(geminiProvider.supports("gemini-1.5-pro", null));
        assertTrue(geminiProvider.supports("gemini-2.0-flash-exp", null));
        assertFalse(geminiProvider.supports("some-other-model", null));
    }

    @Test
    void infer() {
        ProviderRequest request = ProviderRequest.builder()
                .requestId("test-id")
                .model("gemini-1.5-pro")
                .message(Message.user("Hello"))
                .build();

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

        InferenceResponse response = geminiProvider.infer(request).await().indefinitely();

        assertNotNull(response);
        assertEquals("Hello, world!", response.getContent());
        assertEquals(3, response.getMetadata().get("total_tokens"));
    }

    @Test
    void stream() {
        ProviderRequest request = ProviderRequest.builder()
                .requestId("test-id")
                .model("gemini-1.5-pro")
                .message(Message.user("Hello"))
                .build();

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

        List<StreamChunk> chunks = geminiProvider.inferStream(request).collect().asList().await().indefinitely();

        assertNotNull(chunks);
        assertEquals(2, chunks.size());
        assertEquals("Hello, ", chunks.get(0).getDelta());
        assertEquals("world!", chunks.get(1).getDelta());
    }

    @Test
    void health() {
        // Mocking listModels if used, but my implementation simplified it to just
        // return healthy if key present.
        // If I want to test the key presence:
        geminiProvider.initialize(tech.kayys.golek.spi.provider.ProviderConfig.builder("gemini")
                .secret("api.key", "test-key")
                .build());

        assertTrue(geminiProvider.health().await().indefinitely().isHealthy());
    }
}