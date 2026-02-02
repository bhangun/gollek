
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for exception hierarchy.
 */
class ExceptionHierarchyTest {

    @Test
    @DisplayName("InferenceException should store error code and context")
    void testInferenceExceptionCreation() {
        // Given
        String message = "Test error message";
        Map<String, Object> context = Map.of(
                "modelId", "test-model",
                "tenantId", "test-tenant");

        // When
        InferenceException exception = new InferenceException(
                ErrorCode.MODEL_NOT_FOUND,
                message,
                context);

        // Then
        assertEquals(ErrorCode.MODEL_NOT_FOUND, exception.getErrorCode());
        assertEquals(message, exception.getMessage());
        assertEquals(404, exception.getHttpStatusCode());
        assertEquals("test-model", exception.getContext().get("modelId"));
        assertEquals("test-tenant", exception.getContext().get("tenantId"));
        assertFalse(exception.isRetryable());
    }

    @Test
    @DisplayName("InferenceException should support adding context")
    void testAddContext() {
        // Given
        InferenceException exception = new InferenceException(
                ErrorCode.RUNTIME_INFERENCE_FAILED,
                "Inference failed");

        // When
        exception
                .addContext("attempt", 1)
                .addContext("runner", "litert-cpu");

        // Then
        assertEquals(1, exception.getContext().get("attempt"));
        assertEquals("litert-cpu", exception.getContext().get("runner"));
    }

    @Test
    @DisplayName("ModelException should include model ID in context")
    void testModelException() {
        // Given
        String modelId = "gpt-3";

        // When
        ModelException exception = new ModelException(
                ErrorCode.MODEL_NOT_FOUND,
                "Model not found",
                modelId);

        // Then
        assertEquals(modelId, exception.getModelId());
        assertEquals(modelId, exception.getContext().get("modelId"));
        assertEquals(404, exception.getHttpStatusCode());
    }

    @Test
    @DisplayName("TensorException should include tensor name in context")
    void testTensorException() {
        // Given
        String tensorName = "input_ids";

        // When
        TensorException exception = new TensorException(
                ErrorCode.TENSOR_SHAPE_MISMATCH,
                "Shape mismatch",
                tensorName);

        // Then
        assertEquals(tensorName, exception.getTensorName());
        assertEquals(tensorName, exception.getContext().get("tensorName"));
        assertEquals(400, exception.getHttpStatusCode());
    }

    @Test
    @DisplayName("QuotaExceededException should include usage details")
    void testQuotaExceededException() {
        // Given
        String tenantId = "acme-corp";
        String quotaType = "requests";
        long currentUsage = 1000;
        long limit = 1000;

        // When
        QuotaExceededException exception = new QuotaExceededException(
                tenantId,
                quotaType,
                currentUsage,
                limit);

        // Then
        assertEquals(tenantId, exception.getTenantId());
        assertEquals(429, exception.getHttpStatusCode());
        assertEquals(60, exception.getRetryAfterSeconds());
        assertEquals(tenantId, exception.getContext().get("tenantId"));
        assertEquals(quotaType, exception.getContext().get("quotaType"));
        assertEquals(currentUsage, exception.getContext().get("currentUsage"));
        assertEquals(limit, exception.getContext().get("limit"));
    }

    @Test
    @DisplayName("RetryableException should be retryable")
    void testRetryableException() {
        // Given
        RetryableException exception = new RetryableException(
                ErrorCode.DEVICE_OUT_OF_MEMORY,
                "GPU out of memory");

        // Then
        assertTrue(exception.isRetryable());
        assertEquals(3, exception.getMaxRetries());
        assertEquals(100, exception.getBackoffMs());
    }

    @Test
    @DisplayName("ErrorCode should map to correct HTTP status")
    void testErrorCodeHttpMapping() {
        assertEquals(404, ErrorCode.MODEL_NOT_FOUND.getHttpStatus());
        assertEquals(400, ErrorCode.TENSOR_SHAPE_MISMATCH.getHttpStatus());
        assertEquals(429, ErrorCode.QUOTA_EXCEEDED.getHttpStatus());
        assertEquals(401, ErrorCode.AUTH_TOKEN_INVALID.getHttpStatus());
        assertEquals(403, ErrorCode.AUTH_PERMISSION_DENIED.getHttpStatus());
        assertEquals(500, ErrorCode.INIT_RUNNER_FAILED.getHttpStatus());
        assertEquals(503, ErrorCode.DEVICE_NOT_AVAILABLE.getHttpStatus());
        assertEquals(504, ErrorCode.RUNTIME_TIMEOUT.getHttpStatus());
    }

    @Test
    @DisplayName("ErrorCode should identify client vs server errors")
    void testErrorCodeClassification() {
        // Client errors (4xx)
        assertTrue(ErrorCode.MODEL_NOT_FOUND.isClientError());
        assertTrue(ErrorCode.TENSOR_SHAPE_MISMATCH.isClientError());
        assertTrue(ErrorCode.QUOTA_EXCEEDED.isClientError());

        assertFalse(ErrorCode.MODEL_NOT_FOUND.isServerError());

        // Server errors (5xx)
        assertTrue(ErrorCode.INIT_RUNNER_FAILED.isServerError());
        assertTrue(ErrorCode.RUNTIME_INFERENCE_FAILED.isServerError());
        assertTrue(ErrorCode.DEVICE_NOT_AVAILABLE.isServerError());

        assertFalse(ErrorCode.INIT_RUNNER_FAILED.isClientError());
    }

    @Test
    @DisplayName("ErrorCode should identify retryable errors")
    void testRetryableErrors() {
        // Retryable errors
        assertTrue(ErrorCode.DEVICE_OUT_OF_MEMORY.isRetryable());
        assertTrue(ErrorCode.RUNTIME_TIMEOUT.isRetryable());
        assertTrue(ErrorCode.CIRCUIT_BREAKER_OPEN.isRetryable());
        assertTrue(ErrorCode.ALL_RUNNERS_FAILED.isRetryable());

        // Non-retryable errors
        assertFalse(ErrorCode.MODEL_NOT_FOUND.isRetryable());
        assertFalse(ErrorCode.TENSOR_SHAPE_MISMATCH.isRetryable());
        assertFalse(ErrorCode.QUOTA_EXCEEDED.isRetryable());
        assertFalse(ErrorCode.AUTH_TOKEN_INVALID.isRetryable());
    }

    @Test
    @DisplayName("ErrorCode should be findable by code string")
    void testErrorCodeLookup() {
        assertEquals(ErrorCode.MODEL_NOT_FOUND, ErrorCode.fromCode("MODEL_001"));
        assertEquals(ErrorCode.TENSOR_SHAPE_MISMATCH, ErrorCode.fromCode("TENSOR_001"));
        assertEquals(ErrorCode.QUOTA_EXCEEDED, ErrorCode.fromCode("QUOTA_001"));

        // Unknown code should return INTERNAL_ERROR
        assertEquals(ErrorCode.INTERNAL_ERROR, ErrorCode.fromCode("UNKNOWN_999"));
    }

    @Test
    @DisplayName("ErrorResponse should be created from exception")
    void testErrorResponseCreation() {
        // Given
        String requestId = "req-123";
        InferenceException exception = new InferenceException(
                ErrorCode.MODEL_NOT_FOUND,
                "Model not found").addContext("modelId", "gpt-3");

        // When
        ErrorResponse response = ErrorResponse.fromException(exception, requestId);

        // Then
        assertEquals("MODEL_001", response.getErrorCode());
        assertEquals("Model not found", response.getMessage());
        assertEquals(404, response.getHttpStatus());
        assertEquals(requestId, response.getRequestId());
        assertNotNull(response.getTimestamp());
        assertEquals("gpt-3", response.getContext().get("modelId"));
        assertTrue(response.getDocumentationUrl().contains("MODEL_001"));
    }

    @Test
    @DisplayName("Exception chaining should preserve cause")
    void testExceptionChaining() {
        // Given
        Throwable rootCause = new IllegalArgumentException("Root cause");

        // When
        InferenceException exception = new InferenceException(
                ErrorCode.TENSOR_INVALID_DATA,
                "Invalid tensor data",
                rootCause);

        // Then
        assertEquals(rootCause, exception.getCause());
        assertTrue(exception.getMessage().contains("Invalid tensor data"));
    }

    @Test
    @DisplayName("DeviceException should include device type")
    void testDeviceException() {
        // Given
        String deviceType = "cuda";

        // When
        DeviceException exception = new DeviceException(
                ErrorCode.GPU_NOT_FOUND,
                "GPU not found",
                deviceType);

        // Then
        assertEquals(deviceType, exception.getDeviceType());
        assertEquals(deviceType, exception.getContext().get("deviceType"));
    }

    @Test
    @DisplayName("CircuitBreakerOpenException should be retryable")
    void testCircuitBreakerException() {
        // Given
        String runnerName = "litert-cpu";

        // When
        CircuitBreakerOpenException exception = new CircuitBreakerOpenException(runnerName);

        // Then
        assertEquals(runnerName, exception.getRunnerName());
        assertTrue(exception.isRetryable());
        assertEquals(ErrorCode.CIRCUIT_BREAKER_OPEN, exception.getErrorCode());
    }

    @Test
    @DisplayName("toString should provide readable output")
    void testToString() {
        // Given
        InferenceException exception = new InferenceException(
                ErrorCode.MODEL_NOT_FOUND,
                "Test message").addContext("key", "value");

        // When
        String str = exception.toString();

        // Then
        assertTrue(str.contains("MODEL_001"));
        assertTrue(str.contains("404"));
        assertTrue(str.contains("Test message"));
        assertTrue(str.contains("key=value"));
    }
}
