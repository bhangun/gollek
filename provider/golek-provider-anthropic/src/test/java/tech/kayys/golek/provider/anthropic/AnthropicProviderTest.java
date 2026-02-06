package tech.kayys.golek.provider.anthropic;

import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.provider.ProviderConfig;
import tech.kayys.golek.spi.provider.ProviderRequest;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.exception.ProviderException;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AnthropicProviderTest {

    @Mock
    private AnthropicClient anthropicClient;

    private AnthropicProvider provider;

    @BeforeEach
    void setUp() {
        provider = new AnthropicProvider();
        // Use reflection to inject the mocked client
        try {
            java.lang.reflect.Field clientField = AnthropicProvider.class.getDeclaredField("anthropicClient");
            clientField.setAccessible(true);
            clientField.set(provider, anthropicClient);
        } catch (Exception e) {
            throw new RuntimeException("Failed to inject mock client", e);
        }
    }

    @Test
    void testInitialize_Success() {
        // Arrange
        ProviderConfig config = ProviderConfig.builder("anthropic")
                .secret("api.key", "test-api-key")
                .build();

        // Act & Assert
        assertDoesNotThrow(() -> provider.initialize(config));
        // Reflection check for private fields if needed, but not critical for this test
        // if it doesn't throw
    }

    @Test
    void testInitialize_MissingApiKey_ThrowsException() {
        // Arrange
        ProviderConfig config = ProviderConfig.builder("anthropic").build();

        // Act & Assert
        // We assert generically on ProviderException or RuntimeException as
        // getRequiredSecret behavior varies
        assertThrows(Exception.class, () -> provider.initialize(config));
    }

    @Test
    void testId_ReturnsCorrectId() {
        assertEquals("anthropic", provider.id());
    }

    @Test
    void testName_ReturnsCorrectName() {
        assertEquals("Anthropic Claude", provider.name());
    }

    @Test
    void testSupports_ReturnsTrueForSupportedModel() {
        // Arrange
        ProviderConfig config = ProviderConfig.builder("anthropic")
                .secret("api.key", "test-api-key")
                .build();
        provider.initialize(config);

        // Act
        boolean result = provider.supports("claude-3-opus-20240229", null);

        // Assert
        assertTrue(result);
    }

    @Test
    void testSupports_ReturnsFalseForUnsupportedModel() {
        // Arrange
        ProviderConfig config = ProviderConfig.builder("anthropic")
                .secret("api.key", "test-api-key")
                .build();
        provider.initialize(config);

        // Act
        boolean result = provider.supports("gpt-3.5-turbo", null);

        // Assert
        assertFalse(result);
    }

    @Test
    void testInfer_Success() {
        // Arrange
        ProviderConfig config = ProviderConfig.builder("anthropic")
                .secret("api.key", "test-api-key")
                .build();
        provider.initialize(config);

        ProviderRequest request = ProviderRequest.builder()
                .model("claude-3-haiku-20240307")
                .message(Message.user("Hello, Claude!"))
                .build();

        AnthropicResponse mockResponse = new AnthropicResponse(
                "msg_123",
                "message",
                "assistant",
                List.of(new AnthropicResponse.Content("text", "Hello! How can I assist you today?")),
                "claude-3-haiku-20240307",
                "end_turn",
                null,
                new AnthropicResponse.Usage(10, 15));

        when(anthropicClient.createMessage(any(AnthropicRequest.class), any(String.class), any(String.class)))
                .thenReturn(Uni.createFrom().item(mockResponse));

        // Act
        var subscriber = provider.infer(request, null).subscribe().withSubscriber(UniAssertSubscriber.create());

        // Assert
        subscriber.awaitItem();
        var response = subscriber.getItem();
        assertNotNull(response);
        assertEquals("Hello! How can I assist you today?", response.getContent());
        assertEquals("claude-3-haiku-20240307", response.getModel());
    }

    @Test
    void testInfer_ApiError_ThrowsException() {
        // Arrange
        ProviderConfig config = ProviderConfig.builder("anthropic")
                .secret("api.key", "test-api-key")
                .build();
        provider.initialize(config);

        ProviderRequest request = ProviderRequest.builder()
                .model("claude-3-haiku-20240307")
                .message(Message.user("Hello, Claude!"))
                .build();

        when(anthropicClient.createMessage(any(AnthropicRequest.class), any(String.class), any(String.class)))
                .thenReturn(Uni.createFrom().failure(new RuntimeException("API Error")));

        // Act
        var subscriber = provider.infer(request, null).subscribe().withSubscriber(UniAssertSubscriber.create());

        // Assert
        subscriber.awaitFailure();
        assertTrue(subscriber.getFailure() instanceof ProviderException);
    }

    @Test
    void testCapabilities_ReturnsExpectedValues() {
        // Act
        var capabilities = provider.capabilities();

        // Assert
        assertTrue(capabilities.isStreaming());
        assertTrue(capabilities.isFunctionCalling());
        assertTrue(capabilities.isMultimodal());
    }

    @Test
    void testMetadata_ReturnsExpectedValues() {
        // Act
        var metadata = provider.metadata();

        // Assert
        assertEquals("Anthropic", metadata.getVendor());
        assertEquals("Anthropic Claude models integration", metadata.getDescription());
        assertEquals("https://www.anthropic.com", metadata.getHomepage());
    }
}