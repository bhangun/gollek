package tech.kayys.gollek.sdk.core.validation;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.sdk.core.exception.NonRetryableException;

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
        NullPointerException exception = assertThrows(NullPointerException.class,
                () -> InferenceRequest.builder()
                        .messages(List.of(Message.user("Hello")))
                        .build());
        assertTrue(exception.getMessage().contains("model is required"));
    }

    @Test
    void testEmptyMessages() {
        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> InferenceRequest.builder()
                        .model("llama3:latest")
                        .messages(List.of())
                        .build());
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
