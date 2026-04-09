package tech.kayys.gollek.engine.routing;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.kayys.gollek.provider.core.quota.ProviderQuotaService;

import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.routing.ModelProviderMapping;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class MultiProviderRouterAdapterRoutingTest {

    @Mock
    private ProviderRegistry providerRegistry;

    @Mock
    private ProviderQuotaService providerQuotaService;

    private MultiProviderRouter router;
    private ModelProviderRegistry modelProviderRegistry;

    @BeforeEach
    void setUp() {
        router = new MultiProviderRouter();
        modelProviderRegistry = new ModelProviderRegistry();
        modelProviderRegistry.clear();

        router.providerRegistry = providerRegistry;
        router.modelProviderRegistry = modelProviderRegistry;
        router.providerQuotaService = providerQuotaService;
    }

    @Test
    void adapterParameterExcludesAdapterUnsupportedProviders() {
        String modelId = "adapter-model-1";
        registerMapping(modelId, "unsupported-provider", "supported-provider");

        LLMProvider unsupported = provider("unsupported-provider", Set.of("adapter_unsupported"));
        LLMProvider supported = provider("supported-provider", Set.of("adapter_supported"));
        wireProviders(unsupported, supported);

        InferenceRequest request = InferenceRequest.builder()
                .model(modelId)
                .message(Message.user("hello"))
                .parameter("adapter_id", "tenant-a")
                .build();

        RoutingContext context = RoutingContext.simple(request, RequestContext.of("req-1"));
        List<LLMProvider> candidates = router.getCandidates(modelId, context);

        assertEquals(1, candidates.size());
        assertEquals("supported-provider", candidates.get(0).id());
    }

    @Test
    void adapterMetadataExcludesAdapterUnsupportedProviders() {
        String modelId = "adapter-model-2";
        registerMapping(modelId, "unsupported-provider", "supported-provider");

        LLMProvider unsupported = provider("unsupported-provider", Set.of("adapter_unsupported"));
        LLMProvider supported = provider("supported-provider", Set.of("adapter_supported"));
        wireProviders(unsupported, supported);

        InferenceRequest request = InferenceRequest.builder()
                .model(modelId)
                .message(Message.user("hello"))
                .metadata(Map.of("lora_adapter_id", "tenant-b"))
                .build();

        RoutingContext context = RoutingContext.simple(request, RequestContext.of("req-2"));
        List<LLMProvider> candidates = router.getCandidates(modelId, context);

        assertEquals(1, candidates.size());
        assertEquals("supported-provider", candidates.get(0).id());
    }

    @Test
    void nonAdapterRequestKeepsBothProviders() {
        String modelId = "adapter-model-3";
        registerMapping(modelId, "unsupported-provider", "supported-provider");

        LLMProvider unsupported = provider("unsupported-provider", Set.of("adapter_unsupported"));
        LLMProvider supported = provider("supported-provider", Set.of("adapter_supported"));
        wireProviders(unsupported, supported);

        InferenceRequest request = InferenceRequest.builder()
                .model(modelId)
                .message(Message.user("hello"))
                .build();

        RoutingContext context = RoutingContext.simple(request, RequestContext.of("req-3"));
        List<LLMProvider> candidates = router.getCandidates(modelId, context);

        assertEquals(2, candidates.size());
        assertTrue(candidates.stream().anyMatch(p -> p.id().equals("unsupported-provider")));
        assertTrue(candidates.stream().anyMatch(p -> p.id().equals("supported-provider")));
    }

    private void registerMapping(String modelId, String... providerIds) {
        modelProviderRegistry.register(ModelProviderMapping.builder()
                .modelId(modelId)
                .displayName(modelId)
                .providers(providerIds)
                .build());
    }

    private void wireProviders(LLMProvider... providers) {
        for (LLMProvider provider : providers) {
            when(providerRegistry.getProvider(provider.id())).thenReturn(Optional.of(provider));
            when(providerQuotaService.hasQuota(provider.id())).thenReturn(true);
        }
    }

    private static LLMProvider provider(String id, Set<String> features) {
        LLMProvider provider = org.mockito.Mockito.mock(LLMProvider.class);
        ProviderCapabilities capabilities = ProviderCapabilities.builder()
                .features(features)
                .build();
        when(provider.id()).thenReturn(id);
        lenient().when(provider.capabilities()).thenReturn(capabilities);
        when(provider.health()).thenReturn(Uni.createFrom().item(ProviderHealth.healthy()));
        return provider;
    }
}
