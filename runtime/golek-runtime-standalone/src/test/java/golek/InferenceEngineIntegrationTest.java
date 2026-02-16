package tech.kayys.golek.inference.tests.integration;

import tech.kayys.golek.core.inference.InferenceEngine;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.context.RequestContext;

import io.quarkus.test.junit.QuarkusTest;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;

import org.junit.jupiter.api.*;
import jakarta.inject.Inject;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

@QuarkusTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class InferenceEngineIntegrationTest {

        @Inject
        InferenceEngine engine;

        private RequestContext testTenant;

        @BeforeEach
        void setup() {
                testTenant = RequestContext.of("test-tenant");
        }

        @Test
        @Order(1)
        void shouldExecuteValidInferenceRequest() {
                InferenceRequest request = InferenceRequest.builder()
                                .model("gpt-3.5-turbo")
                                .messages(List.of(Message.user("Hello, world!")))
                                .temperature(0.7)
                                .maxTokens(100)
                                .build();

                InferenceResponse response = engine
                                .infer(request, testTenant)
                                .subscribe().withSubscriber(UniAssertSubscriber.create())
                                .awaitItem()
                                .getItem();

                assertThat(response).isNotNull();
                assertThat(response.getContent()).isNotBlank();
                assertThat(response.getTokensUsed()).isGreaterThan(0);
        }

        @Test
        @Order(2)
        void shouldHandleEngineHealth() {
                // Simple health check test
                assertThat(engine.isHealthy()).isTrue();

                var stats = engine.getStats();
                assertThat(stats).isNotNull();
                assertThat(stats.status()).isNotBlank();
        }
}