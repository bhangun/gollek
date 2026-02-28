package tech.kayys.gollek.engine.routing;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.kayys.gollek.engine.model.CachedModelRepository;
import tech.kayys.gollek.engine.observability.RuntimeMetricsCache;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ArtifactLocation;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.exception.NoCompatibleProviderException;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ModelRouterServiceAdapterRoutingTest {

    @Mock
    private ProviderRegistry providerRegistry;

    @Mock
    private CachedModelRepository modelRepository;

    @Mock
    private RuntimeMetricsCache metricsCache;

    private ModelRouterService routerService;

    @BeforeEach
    void setUp() {
        routerService = new ModelRouterService();
        routerService.providerRegistry = providerRegistry;
        routerService.modelRepository = modelRepository;
        routerService.metricsCache = metricsCache;
    }

    @Test
    void adapterRequestSkipsProvidersAdvertisingAdapterUnsupported() {
        ModelManifest manifest = testManifest("model-a");
        InferenceRequest request = InferenceRequest.builder()
                .requestId("req-adapter")
                .model("model-a")
                .message(Message.user("hello"))
                .parameter("adapter_id", "tenant-lora-1")
                .build();

        LLMProvider unsupported = provider("unsupported-provider",
                Set.of("adapter_unsupported"),
                InferenceResponse.builder().requestId("req-adapter").content("unsupported").model("model-a").build());
        LLMProvider supported = provider("supported-provider",
                Set.of("adapter_supported"),
                InferenceResponse.builder().requestId("req-adapter").content("ok").model("model-a").build());

        when(modelRepository.findById("model-a", "community")).thenReturn(Uni.createFrom().item(manifest));
        when(providerRegistry.getAllProviders()).thenReturn(List.of(unsupported, supported));

        InferenceResponse response = routerService.route("model-a", request).await().indefinitely();

        assertEquals("ok", response.getContent());
        verify(unsupported, never()).supports(anyString(), any(ProviderRequest.class));
        verify(unsupported, never()).infer(any(ProviderRequest.class));
        verify(supported).supports(eq("model-a"), any(ProviderRequest.class));
        verify(supported).infer(any(ProviderRequest.class));
    }

    @Test
    void adapterMetadataSkipsProvidersAdvertisingAdapterUnsupported() {
        ModelManifest manifest = testManifest("model-meta");
        InferenceRequest request = InferenceRequest.builder()
                .requestId("req-adapter-meta")
                .model("model-meta")
                .message(Message.user("hello"))
                .metadata(Map.of("adapter_id", "tenant-lora-meta"))
                .build();

        LLMProvider unsupported = provider("unsupported-provider",
                Set.of("adapter_unsupported"),
                InferenceResponse.builder().requestId("req-adapter-meta").content("unsupported").model("model-meta").build());
        LLMProvider supported = provider("supported-provider",
                Set.of("adapter_supported"),
                InferenceResponse.builder().requestId("req-adapter-meta").content("ok-meta").model("model-meta").build());

        when(modelRepository.findById("model-meta", "community")).thenReturn(Uni.createFrom().item(manifest));
        when(providerRegistry.getAllProviders()).thenReturn(List.of(unsupported, supported));

        InferenceResponse response = routerService.route("model-meta", request).await().indefinitely();

        assertEquals("ok-meta", response.getContent());
        verify(unsupported, never()).supports(anyString(), any(ProviderRequest.class));
        verify(unsupported, never()).infer(any(ProviderRequest.class));
        verify(supported).supports(eq("model-meta"), any(ProviderRequest.class));
        verify(supported).infer(any(ProviderRequest.class));
    }

    @Test
    void nonAdapterRequestKeepsProvidersEvenIfTheyAdvertiseAdapterUnsupported() {
        ModelManifest manifest = testManifest("model-b");
        InferenceRequest request = InferenceRequest.builder()
                .requestId("req-no-adapter")
                .model("model-b")
                .message(Message.user("hello"))
                .build();

        LLMProvider unsupported = provider("unsupported-provider",
                Set.of("adapter_unsupported"),
                InferenceResponse.builder().requestId("req-no-adapter").content("plain").model("model-b").build());
        LLMProvider supported = provider("supported-provider",
                Set.of("adapter_supported"),
                InferenceResponse.builder().requestId("req-no-adapter").content("ok").model("model-b").build());

        when(modelRepository.findById("model-b", "community")).thenReturn(Uni.createFrom().item(manifest));
        when(providerRegistry.getAllProviders()).thenReturn(List.of(unsupported, supported));

        InferenceResponse response = routerService.route("model-b", request).await().indefinitely();

        assertEquals("plain", response.getContent());
        verify(unsupported, atLeastOnce()).supports(eq("model-b"), any(ProviderRequest.class));
        verify(unsupported).infer(any(ProviderRequest.class));
    }

    @Test
    void adapterRequestDoesNotUseFallbackForAdapterUnsupportedProvider() {
        ModelManifest manifest = testManifest("model-c");
        InferenceRequest request = InferenceRequest.builder()
                .requestId("req-fallback")
                .model("model-c")
                .message(Message.user("hello"))
                .parameter("adapter_id", "tenant-lora-2")
                .build();

        LLMProvider unsupportedGguf = provider("gguf-provider",
                Set.of("adapter_unsupported"),
                InferenceResponse.builder().requestId("req-fallback").content("should-not-run").model("model-c").build());

        when(modelRepository.findById("model-c", "community")).thenReturn(Uni.createFrom().item(manifest));
        when(providerRegistry.getAllProviders()).thenReturn(List.of(unsupportedGguf));

        assertThrows(NoCompatibleProviderException.class, () -> routerService.route("model-c", request).await().indefinitely());
        verify(unsupportedGguf, never()).infer(any(ProviderRequest.class));
    }

    private static LLMProvider provider(String id, Set<String> features, InferenceResponse response) {
        LLMProvider provider = org.mockito.Mockito.mock(LLMProvider.class);
        ProviderCapabilities capabilities = ProviderCapabilities.builder()
                .supportedFormats(Set.of(ModelFormat.GGUF))
                .features(features)
                .build();

        when(provider.id()).thenReturn(id);
        when(provider.capabilities()).thenReturn(capabilities);
        lenient().when(provider.supports(anyString(), any(ProviderRequest.class))).thenReturn(true);
        lenient().when(provider.infer(any(ProviderRequest.class))).thenReturn(Uni.createFrom().item(response));
        return provider;
    }

    private static ModelManifest testManifest(String modelId) {
        return ModelManifest.builder()
                .modelId(modelId)
                .name(modelId)
                .version("1")
                .path("/tmp/" + modelId + ".gguf")
                .apiKey("community")
                .requestId("community")
                .artifacts(Map.of(ModelFormat.GGUF, new ArtifactLocation("/tmp/" + modelId + ".gguf", null, null, null)))
                .metadata(Map.of())
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();
    }
}
