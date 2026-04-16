package tech.kayys.gollek.plugin.security;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.*;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.execution.ExecutionToken;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;

import java.util.*;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

/**
 * Integration tests for PIIRedactionPlugin.
 * 
 * @since 1.0.0
 */
@QuarkusTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class PIIRedactionPluginIntegrationTest {

    @Inject
    PIIRedactionRequestPlugin requestPlugin;

    @Inject
    PIIRedactionResponsePlugin responsePlugin;

    @Inject
    PIIRedactionService redactionService;

    private ExecutionContext executionContext;
    private EngineContext engineContext;

    @BeforeEach
    void setUp() {
        // Initialize plugin
        var pluginContext = mock(tech.kayys.gollek.spi.plugin.PluginContext.class);
        when(pluginContext.getConfig("enabled", "true")).thenReturn("true");
        when(pluginContext.getConfig("redact-requests", "true")).thenReturn("true");
        when(pluginContext.getConfig("redact-responses", "true")).thenReturn("true");
        when(pluginContext.getConfig("audit-enabled", "true")).thenReturn("true");
        
        requestPlugin.initialize(pluginContext);
        responsePlugin.initialize(pluginContext);
        
        // Mock contexts
        executionContext = mock(ExecutionContext.class);
        engineContext = mock(EngineContext.class);
    }

    @Test
    @Order(1)
    @DisplayName("Should initialize plugins")
    void shouldInitialize() {
        // Then
        assertThat(requestPlugin.id()).isEqualTo("pii-redaction-request");
        assertThat(responsePlugin.id()).isEqualTo("pii-redaction-response");
    }

    @Test
    @Order(2)
    @DisplayName("Should execute in correct phases")
    void shouldExecuteInCorrectPhases() {
        // Then
        assertThat(requestPlugin.phase())
            .isEqualTo(tech.kayys.gollek.spi.inference.InferencePhase.PRE_PROCESSING);
        assertThat(responsePlugin.phase())
            .isEqualTo(tech.kayys.gollek.spi.inference.InferencePhase.POST_PROCESSING);
    }

    @Test
    @Order(3)
    @DisplayName("Should have correct priority order")
    void shouldHaveCorrectPriority() {
        // Then
        assertThat(requestPlugin.order()).isEqualTo(10);
        assertThat(responsePlugin.order()).isEqualTo(10);
    }

    @Test
    @Order(4)
    @DisplayName("Should redact PII from request messages")
    void shouldRedactPIIFromMessages() throws Exception {
        // Given
        List<Message> messages = List.of(
            new Message(
                Message.Role.USER,
                "My email is john.doe@example.com and phone is 555-123-4567"
            )
        );

        InferenceRequest request = InferenceRequest.builder()
            .requestId("test-request-1")
            .model("test-model")
            .messages(messages)
            .build();

        ExecutionToken token = ExecutionToken.builder()
            .requestId("test-request-1")
            .executionId("test-execution-1")
            .build();
        
        when(executionContext.token()).thenReturn(token);
        when(executionContext.getVariable("request", InferenceRequest.class))
            .thenReturn(Optional.of(request));
        when(executionContext.hasError()).thenReturn(false);

        // When
        boolean shouldExecute = requestPlugin.shouldExecute(executionContext);
        assertThat(shouldExecute).isTrue();
        
        requestPlugin.execute(executionContext, engineContext);

        // Then
        verify(executionContext).putVariable(eq("request"), any(InferenceRequest.class));
    }

    @Test
    @Order(5)
    @DisplayName("Should track redaction statistics")
    void shouldTrackStatistics() {
        // Given
        redactionService.clearStats();
        String text = "Contact: test@example.com";

        // When
        redactionService.redact(text);
        var stats = redactionService.getRedactionStats();

        // Then
        assertThat(stats).isNotEmpty();
        assertThat(stats).containsKey("email");
    }

    @Test
    @Order(6)
    @DisplayName("Should detect multiple PII types")
    void shouldDetectMultiplePIITypes() {
        // Given
        String text = "Email: user@example.com, SSN: 123-45-6789, Card: 4532-1234-5678-9012";

        // When
        var detections = redactionService.detectPII(text);

        // Then
        assertThat(detections).hasSizeGreaterThanOrEqualTo(3);
        assertThat(detections).containsKey("email");
        assertThat(detections).containsKey("ssn");
        assertThat(detections).containsKey("credit-card");
    }

    @Test
    @Order(7)
    @DisplayName("Should handle configuration updates")
    void shouldHandleConfigUpdate() {
        // Given
        Map<String, Object> newConfig = new HashMap<>();
        newConfig.put("enabled", false);
        newConfig.put("redact-requests", false);
        newConfig.put("audit-enabled", false);

        // When
        requestPlugin.onConfigUpdate(newConfig);
        responsePlugin.onConfigUpdate(newConfig);
        
        var requestConfig = requestPlugin.currentConfig();
        var responseConfig = responsePlugin.currentConfig();

        // Then
        assertThat(requestConfig).containsEntry("enabled", false);
        assertThat(responseConfig).containsEntry("enabled", false);
    }

    @Test
    @Order(8)
    @DisplayName("Should get pattern information")
    void shouldGetPatternInfo() {
        // When
        var patterns = redactionService.getPatterns();

        // Then
        assertThat(patterns).isNotEmpty();
        assertThat(patterns).containsKey("email");
        assertThat(patterns).containsKey("phone");
        assertThat(patterns).containsKey("credit-card");
    }

    @Test
    @Order(9)
    @DisplayName("Should enable/disable patterns")
    void shouldEnableDisablePatterns() {
        // Given
        redactionService.setPatternEnabled("email", false);

        // When
        var patterns = redactionService.getPatterns();

        // Then
        assertThat(patterns.get("email").enabled).isFalse();

        // Re-enable
        redactionService.setPatternEnabled("email", true);
        assertThat(redactionService.getPatterns().get("email").enabled).isTrue();
    }

    @Test
    @Order(10)
    @DisplayName("Should add custom patterns")
    void shouldAddCustomPattern() {
        // Given
        redactionService.addPattern("custom-test", "\\bCUSTOM\\d+\\b", "[REDACTED_CUSTOM]", true);

        // When
        String result = redactionService.redact("My CUSTOM123 should be hidden");

        // Then
        assertThat(result).contains("[REDACTED_CUSTOM]");
        assertThat(result).doesNotContain("CUSTOM123");
    }

    @Test
    @Order(11)
    @DisplayName("Should handle null request gracefully")
    void shouldHandleNullRequest() throws Exception {
        // Given
        ExecutionToken token = ExecutionToken.builder()
            .requestId("test-request-null")
            .executionId("test-execution-null")
            .build();
        when(executionContext.token()).thenReturn(token);
        when(executionContext.getVariable("request", InferenceRequest.class))
            .thenReturn(Optional.empty());

        // When/Then
        // Should not throw exception
        Assertions.assertDoesNotThrow(() -> {
            requestPlugin.execute(executionContext, engineContext);
        });
    }

    @Test
    @Order(12)
    @DisplayName("Should clear statistics")
    void shouldClearStats() {
        // Given
        redactionService.redact("test@example.com");
        assertThat(redactionService.getRedactionStats()).isNotEmpty();

        // When
        redactionService.clearStats();

        // Then
        assertThat(redactionService.getRedactionStats()).isEmpty();
    }
}
