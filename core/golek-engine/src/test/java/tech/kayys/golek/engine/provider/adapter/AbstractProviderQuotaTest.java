package tech.kayys.golek.engine.provider.adapter;

import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.ProviderCapabilities;
import tech.kayys.golek.spi.provider.ProviderConfig;
import tech.kayys.golek.spi.provider.ProviderHealth;
import tech.kayys.golek.spi.provider.ProviderMetadata;
import tech.kayys.golek.spi.provider.ProviderRequest;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.provider.core.quota.ProviderQuotaService;
import tech.kayys.wayang.tenant.TenantContext;

import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

public class AbstractProviderQuotaTest {

    private TestProvider provider;
    private ProviderQuotaService quotaService;

    @BeforeEach
    void setUp() throws Exception {
        quotaService = mock(ProviderQuotaService.class);
        provider = new TestProvider();
        provider.quotaService = quotaService;

        // Initialize the provider to avoid "Provider not initialized" error
        provider.initialize(ProviderConfig.builder()
                .providerId("test-provider")
                .property("name", "Test Provider")
                .property("version", "1.0.0")
                .property("provider.circuit-breaker.failure-threshold", 5)
                .property("provider.circuit-breaker.timeout", "PT60S")
                .build());

        // Manually inject config properties since we are no running in a container
        java.lang.reflect.Field thresholdField = tech.kayys.golek.engine.provider.adapter.AbstractProvider.class
                .getDeclaredField("circuitBreakerFailureThreshold");
        thresholdField.setAccessible(true);
        thresholdField.setInt(provider, 5);

        java.lang.reflect.Field timeoutField = tech.kayys.golek.engine.provider.adapter.AbstractProvider.class
                .getDeclaredField("circuitBreakerTimeout");
        timeoutField.setAccessible(true);
        timeoutField.set(provider, java.time.Duration.ofSeconds(60));
    }

    @Test
    void shouldAllowInferenceWhenQuotaAvailable() {
        // Arrange
        when(quotaService.hasQuota(any())).thenReturn(true);
        ProviderRequest request = ProviderRequest.builder()
                .model("test-model")
                .requestId("test-req-id")
                .message(Message.user("hello"))
                .build();
        TenantContext context = TenantContext.of("tenant-1");

        // Act
        UniAssertSubscriber<InferenceResponse> subscriber = provider.infer(request, context)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem();

        InferenceResponse response = subscriber.getItem();
        Assertions.assertEquals("test-response", response.getContent());

        // Assert
        verify(quotaService).hasQuota(eq("test-provider"));
        verify(quotaService).recordUsage(eq("test-provider"), anyInt());
    }

    @Test
    void shouldDenyInferenceWhenQuotaExhausted() {
        // Arrange
        when(quotaService.hasQuota(any())).thenReturn(false);
        ProviderRequest request = ProviderRequest.builder()
                .model("test-model")
                .requestId("test-req-id")
                .message(Message.user("hello"))
                .build();
        TenantContext context = TenantContext.of("tenant-1");

        // Act & Assert
        provider.infer(request, context)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure()
                .assertFailedWith(tech.kayys.golek.spi.routing.QuotaExhaustedException.class);

        verify(quotaService).hasQuota(eq("test-provider"));
        verify(quotaService, never()).recordUsage(any(), anyInt());
    }

    private static class TestProvider extends AbstractProvider {
        @Override
        public String id() {
            return "test-provider";
        }

        @Override
        public String version() {
            return "1.0.0";
        }

        @Override
        public String name() {
            return "Test Provider";
        }

        @Override
        public ProviderMetadata metadata() {
            return null;
        }

        @Override
        public ProviderCapabilities capabilities() {
            return null;
        }

        @Override
        public boolean supports(String modelId, TenantContext tenantContext) {
            return true;
        }

        @Override
        protected Uni<Void> doInitialize(Map<String, Object> config, TenantContext tenant) {
            return Uni.createFrom().voidItem();
        }

        @Override
        protected Uni<InferenceResponse> doInfer(ProviderRequest request) {
            return Uni.createFrom().item(InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .content("test-response")
                    .tokensUsed(10)
                    .build());
        }

        @Override
        protected Uni<ProviderHealth> doHealthCheck() {
            return Uni.createFrom().item(ProviderHealth.healthy("OK"));
        }
    }
}
