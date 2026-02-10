package tech.kayys.golek.sdk.core.validation;

import org.junit.jupiter.api.Test;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.sdk.core.exception.NonRetryableException;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RequestValidatorTest {

    @Test
    void testValidRequest() {
        InferenceRequest request = InferenceRequest.builder()
                .model("llama3:latest")
                .messages(List.of(Message.user("Hello")))
                .temperature(0.7)
                .maxTokens(100)
                .build();

        assertDoesNotThrow(() -> RequestValidator.validate(request));
    }

    @Test
    void testNullRequest() {
        NonRetryableException exception = assertThrows(NonRetryableException.class,
                () -> RequestValidator.validate(null));
        assertTrue(exception.getMessage().contains("cannot be null"));
    }

    @Test
    void testMissingModel() {
        InferenceRequest request = InferenceRequest.builder()
                .messages(List.of(Message.user("Hello")))
                .build();

        NonRetryableException exception = assertThrows(NonRetryableException.class,
                () -> RequestValidator.validate(request));
        assertTrue(exception.getMessage().contains("Model name is required"));
    }

    @Test
    void testEmptyMessages() {
        InferenceRequest request = InferenceRequest.builder()
                .model("llama3:latest")
                .messages(List.of())
                .build();

        NonRetryableException exception = assertThrows(NonRetryableException.class,
                () -> RequestValidator.validate(request));
        assertTrue(exception.getMessage().contains("At least one message is required"));
    }

    @Test
    void testInvalidTemperature() {
        InferenceRequest request = InferenceRequest.builder()
                .model("llama3:latest")
                .messages(List.of(Message.user("Hello")))
                .temperature(3.0) // Invalid: > 2.0
                .build();

        NonRetryableException exception = assertThrows(NonRetryableException.class,
                () -> RequestValidator.validate(request));
        assertTrue(exception.getMessage().contains("Temperature must be between"));
    }

    @Test
    void testInvalidMaxTokens() {
        InferenceRequest request = InferenceRequest.builder()
                .model("llama3:latest")
                .messages(List.of(Message.user("Hello")))
                .maxTokens(-1) // Invalid: negative
                .build();

        NonRetryableException exception = assertThrows(NonRetryableException.class,
                () -> RequestValidator.validate(request));
        assertTrue(exception.getMessage().contains("Max tokens must be positive"));
    }
}
