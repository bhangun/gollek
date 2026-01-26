package tech.kayys.golek.inference.tests.integration;

import tech.kayys.golek.inference.kernel.engine.*;
import tech.kayys.golek.inference.kernel.plugin.*;
import tech.kayys.golek.inference.api.*;

import io.quarkus.test.junit.QuarkusTest;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;

import org.junit.jupiter.api.*;
import jakarta.inject.Inject;

import static org.assertj.core.api.Assertions.*;

@QuarkusTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class InferenceEngineIntegrationTest {

    @Inject
    InferenceEngine engine;

    @Inject
    GolekPluginRegistry pluginRegistry;

    private TenantContext testTenant;

    @BeforeEach
    void setup() {
        testTenant = TenantContext.of(TenantId.of("test-tenant"));
    }

    @Test
    @Order(1)
    void shouldLoadAllPlugins() {
        List<GolekPlugin> plugins = pluginRegistry.all();

        assertThat(plugins)
                .isNotEmpty()
                .anyMatch(p -> p.descriptor().type() == PluginType.VALIDATION)
                .anyMatch(p -> p.descriptor().type() == PluginType.POLICY);
    }

    @Test
    @Order(2)
    void shouldExecuteValidInferenceRequest() {
        InferenceRequest request = InferenceRequest.builder()
                .model("gpt-3.5-turbo")
                .message(Message.user("Hello, world!"))
                .temperature(0.7)
                .maxTokens(100)
                .build();

        InferenceResponse response = engine
                .infer(request, testTenant)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem()
                .getItem();

        assertThat(response).isNotNull();
        assertThat(response.content()).isNotBlank();
        assertThat(response.tokensUsed()).isGreaterThan(0);
    }

    @Test
    @Order(3)
    void shouldRejectUnsafeContent() {
        InferenceRequest request = InferenceRequest.builder()
                .model("gpt-3.5-turbo")
                .message(Message.user("How to hack a system?"))
                .build();

        assertThatThrownBy(() -> engine.infer(request, testTenant)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure()
                .getFailure()).isInstanceOf(ContentSafetyException.class);
    }

    @Test
    @Order(4)
    void shouldEnforceQuota() {
        // Simulate quota exhaustion
        TenantContext quotaExhausted = TenantContext.of(
                TenantId.of("quota-exhausted-tenant"));

        InferenceRequest request = InferenceRequest.builder()
                .model("gpt-3.5-turbo")
                .message(Message.user("Test"))
                .build();

        assertThatThrownBy(() -> engine.infer(request, quotaExhausted)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitFailure()
                .getFailure()).isInstanceOf(TenantQuotaExceededException.class);
    }

    @Test
    @Order(5)
    void shouldFallbackOnProviderFailure() {
        // Mock primary provider failure
        InferenceRequest request = InferenceRequest.builder()
                .model("unavailable-model")
                .message(Message.user("Test"))
                .preferredProvider("failing-provider")
                .build();

        // Should fallback to alternative provider
        InferenceResponse response = engine
                .infer(request, testTenant)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem()
                .getItem();

        assertThat(response).isNotNull();
        assertThat(response.metadata().get("provider"))
                .isNotEqualTo("failing-provider");
    }
}