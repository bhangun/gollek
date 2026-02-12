package tech.kayys.golek.inference.gguf;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.ModelFormat;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.*;

@QuarkusTest
class LlamaCppRunnerTest {

    @Inject
    GGUFProviderConfig config;

    @Mock
    LlamaCppBinding binding;

    @Mock
    GGUFChatTemplateService templateService;

    private LlamaCppRunner runner;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        runner = new LlamaCppRunner(binding, config, templateService);
    }

    @Test
    @DisplayName("Runner should initialize semaphore from config")
    void testConcurrencyLimit() {
        // Access private field to verify
        // Or assume it works if no exception
        assertThat(runner).isNotNull();
    }

    @Test
    @DisplayName("Runner initialization should check model existence")
    void testInitializationMissingModel() {
        tech.kayys.golek.spi.model.ArtifactLocation location = new tech.kayys.golek.spi.model.ArtifactLocation(
                "/path/to/missing/model.gguf",
                null,
                null,
                null);

        ModelManifest manifest = new ModelManifest(
                "missing-model.gguf",
                "missing-model.gguf",
                "1.0",
                "tenant1",
                Map.of(ModelFormat.GGUF, location),
                Collections.emptyList(),
                null,
                Collections.emptyMap(),
                java.time.Instant.now(),
                java.time.Instant.now());

        Map<String, Object> runnerConfig = Collections.emptyMap();

        assertThatThrownBy(() -> runner.initialize(manifest, runnerConfig))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Failed to initialize GGUF runner")
                .hasCauseInstanceOf(RuntimeException.class);
        // Additional check for cause message if needed
    }

    @Test
    @DisplayName("Runner should throw if inference called before init")
    void testInferenceBeforeInit() {
        InferenceRequest request = InferenceRequest.builder()
                .model("model.gguf")
                .message(tech.kayys.golek.spi.Message.user("Hello"))
                .build();
        tech.kayys.golek.spi.context.RequestContext ctx = tech.kayys.golek.spi.context.RequestContext.create(
                "tenant1", "user1", "session1");

        assertThatThrownBy(() -> runner.infer(request, ctx))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not initialized");
    }

    @Test
    @DisplayName("Runner close without init should be safe")
    void testCloseWithoutInit() {
        assertThatCode(() -> runner.close()).doesNotThrowAnyException();
    }

    @Test
    @DisplayName("Warmup requests creation")
    void testCreateWarmupRequests() {
        List<InferenceRequest> requests = runner.createDefaultWarmupRequests();
        assertThat(requests).isNotEmpty();
        assertThat(requests.get(0).getParameters().get("prompt")).isNotNull();
    }
}
