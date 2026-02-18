package tech.kayys.gollek.engine.inference;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.kayys.gollek.engine.observability.MetricsPublisher;
import tech.kayys.gollek.engine.routing.ModelRouterService;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.InferenceStage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class StageAwareOrchestratorTest {

    @Mock
    private ModelRouterService router;

    @Mock
    private MetricsPublisher metrics;

    private InferenceOrchestrator orchestrator;

    @BeforeEach
    void setUp() {
        orchestrator = new InferenceOrchestrator(router, metrics);
        // Default: disaggregated mode disabled
        orchestrator.setDisaggregatedMode(false);
    }

    @Test
    void whenDisaggregatedModeDisabled_stagesResolveToCombined() {
        // Arrange
        InferenceRequest request = InferenceRequest.builder()
                .model("llama-3-8b")
                .message(Message.user("Short generic prompt"))
                .build();

        when(router.route(any(), any())).thenReturn(Uni.createFrom().item(mock(InferenceResponse.class)));

        // Act
        orchestrator.executeAsync("llama-3-8b", request).await().indefinitely();

        // Assert
        ArgumentCaptor<InferenceRequest> captor = ArgumentCaptor.forClass(InferenceRequest.class);
        verify(router).route(eq("llama-3-8b"), captor.capture());

        InferenceRequest routedRequest = captor.getValue();
        assertThat(routedRequest.getInferenceStage()).isEqualTo(InferenceStage.COMBINED);
    }

    @Test
    void whenDisaggregatedModeEnabled_smallPrompt_resolvesToCombined() {
        // Arrange
        orchestrator.setDisaggregatedMode(true);
        orchestrator.setSmallPromptThreshold(100);

        // "Short prompt" is ~12 chars -> ~3 tokens << 100
        InferenceRequest request = InferenceRequest.builder()
                .model("llama-3-8b")
                .message(Message.user("Short prompt"))
                .build();

        when(router.route(any(), any())).thenReturn(Uni.createFrom().item(mock(InferenceResponse.class)));

        // Act
        orchestrator.executeAsync("llama-3-8b", request).await().indefinitely();

        // Assert
        ArgumentCaptor<InferenceRequest> captor = ArgumentCaptor.forClass(InferenceRequest.class);
        verify(router).route(eq("llama-3-8b"), captor.capture());

        assertThat(captor.getValue().getInferenceStage()).isEqualTo(InferenceStage.COMBINED);
    }

    @Test
    void whenDisaggregatedModeEnabled_largePrompt_resolvesToPrefill() {
        // Arrange
        orchestrator.setDisaggregatedMode(true);
        orchestrator.setSmallPromptThreshold(5);

        // "This is a longer prompt that should trigger prefill stage"
        // 60 chars -> ~15 tokens > 5
        String longText = "This is a longer prompt that should trigger prefill stage routing logic.";
        InferenceRequest request = InferenceRequest.builder()
                .model("llama-3-8b")
                .message(Message.user(longText))
                .build();

        when(router.route(any(), any())).thenReturn(Uni.createFrom().item(mock(InferenceResponse.class)));

        // Act
        orchestrator.executeAsync("llama-3-8b", request).await().indefinitely();

        // Assert
        ArgumentCaptor<InferenceRequest> captor = ArgumentCaptor.forClass(InferenceRequest.class);
        verify(router).route(eq("llama-3-8b"), captor.capture());

        InferenceRequest routed = captor.getValue();
        assertThat(routed.getInferenceStage()).isEqualTo(InferenceStage.PREFILL);
        assertThat(routed.getPromptTokenCount()).isGreaterThan(5);
    }

    @Test
    void explicitStageIsRespected() {
        // Arrange
        orchestrator.setDisaggregatedMode(true);

        // Explicitly set DECODE stage even if it looks like a prefill
        InferenceRequest request = InferenceRequest.builder()
                .model("llama-3-8b")
                .message(Message.user("Some prompt"))
                .inferenceStage(InferenceStage.DECODE)
                .build();

        when(router.route(any(), any())).thenReturn(Uni.createFrom().item(mock(InferenceResponse.class)));

        // Act
        orchestrator.executeAsync("llama-3-8b", request).await().indefinitely();

        // Assert
        ArgumentCaptor<InferenceRequest> captor = ArgumentCaptor.forClass(InferenceRequest.class);
        verify(router).route(eq("llama-3-8b"), captor.capture());

        assertThat(captor.getValue().getInferenceStage()).isEqualTo(InferenceStage.DECODE);
    }
}
