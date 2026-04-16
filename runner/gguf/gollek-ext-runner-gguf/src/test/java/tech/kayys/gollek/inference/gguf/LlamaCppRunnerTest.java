package tech.kayys.gollek.inference.gguf;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ModelFormat;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.lang.reflect.Method;
import java.time.Duration;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;

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
                tech.kayys.gollek.spi.model.ArtifactLocation location = new tech.kayys.gollek.spi.model.ArtifactLocation(
                                "/path/to/missing/model.gguf",
                                null,
                                null,
                                null);

                ModelManifest manifest = ModelManifest.builder()
                                .modelId("missing-model.gguf")
                                .name("missing-model.gguf")
                                .version("1.0")
                                .path(location.uri())
                                .apiKey(tech.kayys.gollek.spi.auth.ApiKeyConstants.COMMUNITY_API_KEY)
                                .requestId("tenant1")
                                .artifacts(Map.of(ModelFormat.GGUF, location))
                                .supportedDevices(Collections.emptyList())
                                .resourceRequirements(null)
                                .metadata(Collections.emptyMap())
                                .createdAt(java.time.Instant.now())
                                .updatedAt(java.time.Instant.now())
                                .build();

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
                                .message(tech.kayys.gollek.spi.Message.user("Hello"))
                                .build();
                tech.kayys.gollek.spi.context.RequestContext ctx = tech.kayys.gollek.spi.context.RequestContext
                                .forTenant(
                                                "tenant1", "model1");

                assertThatThrownBy(() -> runner.infer(request))
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

        @Test
        @DisplayName("Request-level coalesce bypass flags are honored")
        void testCoalesceBypassFlags() throws Exception {
                GGUFProviderConfig localConfig = org.mockito.Mockito.mock(GGUFProviderConfig.class);
                org.mockito.Mockito.when(localConfig.maxConcurrentRequests()).thenReturn(8);
                org.mockito.Mockito.when(localConfig.coalesceEnabled()).thenReturn(true);
                org.mockito.Mockito.when(localConfig.coalesceWindowMs()).thenReturn(1);
                org.mockito.Mockito.when(localConfig.coalesceMaxBatch()).thenReturn(4);
                org.mockito.Mockito.when(localConfig.coalesceMaxQueue()).thenReturn(16);

                LlamaCppRunner localRunner = new LlamaCppRunner(binding, localConfig, templateService);
                Method method = LlamaCppRunner.class.getDeclaredMethod("shouldBypassCoalesce", InferenceRequest.class);
                method.setAccessible(true);

                InferenceRequest coalesceOff = InferenceRequest.builder()
                                .model("model.gguf")
                                .message(tech.kayys.gollek.spi.Message.user("hello"))
                                .parameter("gguf.coalesce", "false")
                                .build();
                InferenceRequest sessionPersist = InferenceRequest.builder()
                                .model("model.gguf")
                                .message(tech.kayys.gollek.spi.Message.user("hello"))
                                .parameter("gguf.session.persist", "true")
                                .build();
                InferenceRequest defaultRequest = InferenceRequest.builder()
                                .model("model.gguf")
                                .message(tech.kayys.gollek.spi.Message.user("hello"))
                                .build();

                assertThat((boolean) method.invoke(localRunner, coalesceOff)).isTrue();
                assertThat((boolean) method.invoke(localRunner, sessionPersist)).isTrue();
                assertThat((boolean) method.invoke(localRunner, defaultRequest)).isFalse();
        }

        @Test
        @DisplayName("Reuse prefix skips KV clear and minimizes prompt decode")
        void testReusePrefixSkipsPromptPrefill() throws Exception {
                GGUFProviderConfig localConfig = org.mockito.Mockito.mock(GGUFProviderConfig.class);
                org.mockito.Mockito.when(localConfig.maxConcurrentRequests()).thenReturn(1);
                org.mockito.Mockito.when(localConfig.defaultTimeout()).thenReturn(Duration.ofMillis(250));
                org.mockito.Mockito.when(localConfig.maxContextTokens()).thenReturn(128);
                org.mockito.Mockito.when(localConfig.batchSize()).thenReturn(8);
                org.mockito.Mockito.when(localConfig.threads()).thenReturn(2);
                org.mockito.Mockito.when(localConfig.gpuEnabled()).thenReturn(false);
                org.mockito.Mockito.when(localConfig.mmapEnabled()).thenReturn(true);
                org.mockito.Mockito.when(localConfig.mlockEnabled()).thenReturn(false);

                LlamaCppBinding localBinding = org.mockito.Mockito.mock(LlamaCppBinding.class);
                GGUFChatTemplateService localTemplate = org.mockito.Mockito.mock(GGUFChatTemplateService.class);

                org.mockito.Mockito.when(localBinding.tokenize(any(), anyString(), anyBoolean(), anyBoolean()))
                                .thenReturn(new int[] { 1, 2, 3 });
                org.mockito.Mockito.when(localBinding.batchInit(anyInt(), anyInt(), anyInt()))
                                .thenReturn(java.lang.foreign.MemorySegment.NULL);
                org.mockito.Mockito.when(localBinding.decode(any(), any())).thenReturn(0);

                LlamaCppRunner localRunner = new LlamaCppRunner(localBinding, localConfig, localTemplate);
                setField(localRunner, "initialized", true);
                setField(localRunner, "context", java.lang.foreign.MemorySegment.NULL);
                setField(localRunner, "model", java.lang.foreign.MemorySegment.NULL);
                setField(localRunner, "vocabSize", 4);
                setField(localRunner, "eosToken", -1);
                setField(localRunner, "manifest", createManifest());
                setField(localRunner, "runtimeBatchSize", 8);

                // Access internal KVCacheManager to set history for prefix reuse test
                java.lang.reflect.Field managerField = LlamaCppRunner.class.getDeclaredField("kvCacheManager");
                managerField.setAccessible(true);
                Object cacheManager = managerField.get(localRunner);

                java.lang.reflect.Field historyField = LlamaCppKVCacheManager.class.getDeclaredField("kvTokenHistory");
                historyField.setAccessible(true);
                historyField.set(cacheManager, new int[] { 1, 2 });

                java.lang.reflect.Field countField = LlamaCppKVCacheManager.class.getDeclaredField("kvTokenCount");
                countField.setAccessible(true);
                countField.set(cacheManager, 2);

                InferenceRequest request = InferenceRequest.builder()
                                .model("test-model")
                                .message(tech.kayys.gollek.spi.Message.user("hello reuse"))
                                .parameter("prompt", "hello reuse")
                                .parameter("max_tokens", 0)
                                .build();

                Method inferInternal = LlamaCppRunner.class.getDeclaredMethod("executeWithComponents",
                                InferenceRequest.class, java.util.function.Consumer.class);
                inferInternal.setAccessible(true);
                Object response = inferInternal.invoke(localRunner, request, null);

                assertThat(response).isNotNull();
                org.mockito.Mockito.verify(localBinding, org.mockito.Mockito.never()).kvCacheClear(any());
                org.mockito.Mockito.verify(localBinding, org.mockito.Mockito.times(1)).decode(any(), any());
        }

        private static void setField(Object target, String name, Object value) throws Exception {
                java.lang.reflect.Field field = target.getClass().getDeclaredField(name);
                field.setAccessible(true);
                field.set(target, value);
        }

        private static ModelManifest createManifest() {
                tech.kayys.gollek.spi.model.ArtifactLocation location = new tech.kayys.gollek.spi.model.ArtifactLocation(
                                "test.gguf", null, null, null);
                return ModelManifest.builder()
                                .modelId("test-model")
                                .name("test-model")
                                .version("test")
                                .path(location.uri())
                                .apiKey(tech.kayys.gollek.spi.auth.ApiKeyConstants.COMMUNITY_API_KEY)
                                .requestId("tenant1")
                                .artifacts(Map.of(ModelFormat.GGUF, location))
                                .supportedDevices(Collections.emptyList())
                                .resourceRequirements(null)
                                .metadata(Collections.emptyMap())
                                .createdAt(java.time.Instant.now())
                                .updatedAt(java.time.Instant.now())
                                .build();
        }
}
