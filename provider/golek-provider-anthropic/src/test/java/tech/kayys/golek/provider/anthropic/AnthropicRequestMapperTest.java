package tech.kayys.golek.provider.anthropic;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.provider.ProviderRequest;

class AnthropicRequestMapperTest {

    @Test
    void testMapToAnthropicRequest_BasicMapping() {
        // Arrange
        ProviderRequest providerRequest = ProviderRequest.builder()
                .model("claude-3-haiku-20240307")
                .message(Message.user("Hello, Claude!"))
                .parameter("temperature", 0.7)
                .parameter("max_tokens", 100)
                .build();

        AnthropicProvider provider = new AnthropicProvider();

        // Use reflection to access the private method
        try {
            java.lang.reflect.Method method = AnthropicProvider.class.getDeclaredMethod("mapToAnthropicRequest",
                    ProviderRequest.class);
            method.setAccessible(true);

            // Act
            AnthropicRequest result = (AnthropicRequest) method.invoke(provider, providerRequest);

            // Assert
            assertEquals("claude-3-haiku-20240307", result.model());
            assertEquals(100, result.maxTokens());
            assertEquals(0.7, result.temperature());
            assertEquals(1, result.messages().size());
            assertEquals("user", result.messages().get(0).role());
            assertEquals("Hello, Claude!", result.messages().get(0).content());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void testMapToAnthropicRequest_WithSystemMessage() {
        // Arrange
        ProviderRequest providerRequest = ProviderRequest.builder()
                .model("claude-3-haiku-20240307")
                .message(Message.system("You are a helpful assistant."))
                .message(Message.user("Hello, Claude!"))
                .parameter("temperature", 0.5)
                .parameter("max_tokens", 200)
                .build();

        AnthropicProvider provider = new AnthropicProvider();

        // Use reflection to access the private method
        try {
            java.lang.reflect.Method method = AnthropicProvider.class.getDeclaredMethod("mapToAnthropicRequest",
                    ProviderRequest.class);
            method.setAccessible(true);

            // Act
            AnthropicRequest result = (AnthropicRequest) method.invoke(provider, providerRequest);

            // Assert
            assertEquals("claude-3-haiku-20240307", result.model());
            assertEquals(200, result.maxTokens());
            assertEquals(0.5, result.temperature());
            assertEquals(1, result.messages().size()); // System message should be extracted
            assertEquals("user", result.messages().get(0).role());
            assertEquals("Hello, Claude!", result.messages().get(0).content());
            assertEquals("You are a helpful assistant.", result.systemPrompt());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void testMapToAnthropicRequest_MultipleMessages() {
        // Arrange
        ProviderRequest providerRequest = ProviderRequest.builder()
                .model("claude-3-haiku-20240307")
                .message(Message.user("Hello, Claude!"))
                .message(Message.assistant("Hello! How can I help you?"))
                .message(Message.user("Can you explain quantum computing?"))
                .parameter("temperature", 0.8)
                .parameter("max_tokens", 500)
                .build();

        AnthropicProvider provider = new AnthropicProvider();

        // Use reflection to access the private method
        try {
            java.lang.reflect.Method method = AnthropicProvider.class.getDeclaredMethod("mapToAnthropicRequest",
                    ProviderRequest.class);
            method.setAccessible(true);

            // Act
            AnthropicRequest result = (AnthropicRequest) method.invoke(provider, providerRequest);

            // Assert
            assertEquals("claude-3-haiku-20240307", result.model());
            assertEquals(500, result.maxTokens());
            assertEquals(0.8, result.temperature());
            assertEquals(3, result.messages().size());
            assertEquals("user", result.messages().get(0).role());
            assertEquals("Hello, Claude!", result.messages().get(0).content());
            assertEquals("assistant", result.messages().get(1).role());
            assertEquals("Hello! How can I help you?", result.messages().get(1).content());
            assertEquals("user", result.messages().get(2).role());
            assertEquals("Can you explain quantum computing?", result.messages().get(2).content());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}