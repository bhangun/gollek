package tech.kayys.golek.engine.inference;

import lombok.Builder;
import lombok.Data;
import tech.kayys.golek.api.exception.InferenceException;

import java.time.Instant;

/**
 * Standardized error response for the inference API.
 */
@Data
@Builder
public class ErrorResponse {
    private String errorCode;
    private String message;
    private int httpStatus;
    private String requestId;
    private Instant timestamp;

    public static ErrorResponse fromException(InferenceException ex, String requestId) {
        return ErrorResponse.builder()
                .errorCode(ex.getErrorCode() != null ? ex.getErrorCode().getCode() : "INTERNAL_ERROR")
                .message(ex.getMessage())
                .httpStatus(ex.getHttpStatusCode())
                .requestId(requestId)
                .timestamp(Instant.now())
                .build();
    }
}
