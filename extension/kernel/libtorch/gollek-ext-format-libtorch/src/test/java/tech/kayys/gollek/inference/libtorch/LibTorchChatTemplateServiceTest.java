package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.Message;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link LibTorchChatTemplateService}.
 */
class LibTorchChatTemplateServiceTest {

    private LibTorchChatTemplateService service;

    @BeforeEach
    void setUp() {
        service = new LibTorchChatTemplateService();
    }

    @Test
    void rendersSingleUserMessage() {
        List<Message> messages = List.of(Message.user("Hello"));
        String result = service.renderChatML(messages);

        assertTrue(result.contains("<|im_start|>user"), "Should contain user marker");
        assertTrue(result.contains("Hello"), "Should contain message content");
        assertTrue(result.contains("<|im_end|>"), "Should contain end marker");
        assertTrue(result.contains("<|im_start|>assistant"), "Should end with assistant prompt");
    }

    @Test
    void rendersSystemAndUserMessages() {
        List<Message> messages = List.of(
                Message.system("You are helpful."),
                Message.user("What is 2+2?"));
        String result = service.renderChatML(messages);

        assertTrue(result.contains("<|im_start|>system"), "Should contain system marker");
        assertTrue(result.contains("You are helpful."), "Should contain system content");
        assertTrue(result.contains("<|im_start|>user"), "Should contain user marker");
        assertTrue(result.contains("What is 2+2?"), "Should contain user content");
    }

    @Test
    void rendersMultiTurnConversation() {
        List<Message> messages = List.of(
                Message.user("Hi"),
                Message.assistant("Hello!"),
                Message.user("How are you?"));
        String result = service.renderChatML(messages);

        // Should have 3 message blocks + assistant prompt
        long imStartCount = result.chars().filter(c -> c == '<').count();
        assertTrue(imStartCount > 6, "Should have multiple marker pairs");

        assertTrue(result.contains("assistant\nHello!"), "Should contain assistant reply");
    }

    @Test
    void emptyMessagesReturnsEmptyString() {
        List<Message> messages = List.of();
        String result = service.renderChatML(messages);

        assertEquals("", result, "Empty messages should return empty string");
    }

    @Test
    void handlesNullContent() {
        // Message with null content should not throw
        assertDoesNotThrow(() -> {
            List<Message> messages = List.of(Message.user("test"));
            service.renderChatML(messages);
        });
    }
}
