package tech.kayys.golek.client.builder;

import org.junit.jupiter.api.Test;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.inference.InferenceRequest;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class InferenceRequestTest {

    @Test
    void testBuildInferenceRequest() {
        // Act
        InferenceRequest request = InferenceRequest.builder()
                .model("test-model")
                .message(Message.system("System message"))
                .message(Message.user("Hello, world!"))
                .temperature(0.7)
                .maxTokens(100)
                .topP(0.9)
                .parameter("custom_param", "custom_value")
                .build();

        // Assert
        assertNotNull(request);
        assertEquals("test-model", request.getModel());
        assertEquals(2, request.getMessages().size());
        assertEquals("Hello, world!", request.getMessages().get(1).getContent());
        assertEquals("System message", request.getMessages().get(0).getContent());
        assertEquals(0.7, request.getParameters().get("temperature"));
        assertEquals(100, request.getParameters().get("max_tokens"));
        assertEquals(0.9, request.getParameters().get("top_p"));
        assertEquals("custom_value", request.getParameters().get("custom_param"));
    }

    @Test
    void testBuildInferenceRequestWithAllFields() {
        // Act
        InferenceRequest request = InferenceRequest.builder()
                .requestId("custom-request-id")
                .model("test-model")
                .message(Message.user("Test message"))
                .parameter("param1", "value1")
                .parameters(Map.of("param2", "value2"))
                .temperature(0.5)
                .maxTokens(200)
                .topP(0.8)
                .streaming(true)
                .preferredProvider("specific-provider")
                .priority(8)
                .build();

        // Assert
        assertNotNull(request);
        assertEquals("custom-request-id", request.getRequestId());
        assertEquals("test-model", request.getModel());
        assertEquals(1, request.getMessages().size());
        assertEquals("Test message", request.getMessages().get(0).getContent());
        assertEquals(0.5, request.getParameters().get("temperature"));
        assertEquals(200, request.getParameters().get("max_tokens"));
        assertEquals(0.8, request.getParameters().get("top_p"));
        assertEquals("value1", request.getParameters().get("param1"));
        assertEquals("value2", request.getParameters().get("param2"));
        assertTrue(request.isStreaming());
        assertEquals("specific-provider", request.getPreferredProvider().orElse(null));
        assertEquals(8, request.getPriority());
    }
}

class MessageBuilderTest {

    @Test
    void testBuildMessage() {
        // Act
        Message message = new Message(Message.Role.USER, "Test content", "test-user", null, null);

        // Assert
        assertNotNull(message);
        assertEquals(Message.Role.USER, message.getRole());
        assertEquals("Test content", message.getContent());
        assertEquals("test-user", message.getName());
    }

    @Test
    void testConvenienceMethods() {
        // Act
        Message systemMsg = Message.system("System message");
        Message userMsg = Message.user("User message");
        Message assistantMsg = Message.assistant("Assistant message");
        Message toolMsg = Message.tool("tool-call-id", "Tool response");

        // Assert
        assertEquals(Message.Role.SYSTEM, systemMsg.getRole());
        assertEquals("System message", systemMsg.getContent());

        assertEquals(Message.Role.USER, userMsg.getRole());
        assertEquals("User message", userMsg.getContent());

        assertEquals(Message.Role.ASSISTANT, assistantMsg.getRole());
        assertEquals("Assistant message", assistantMsg.getContent());

        assertEquals(Message.Role.TOOL, toolMsg.getRole());
        assertEquals("Tool response", toolMsg.getContent());
        assertEquals("tool-call-id", toolMsg.getToolCallId());
    }
}