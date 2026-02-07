package tech.kayys.golek.sdk.core.validation;

import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.sdk.core.exception.NonRetryableException;

/**
 * Validator for inference requests.
 */
public class RequestValidator {
    
    /**
     * Validates an inference request.
     * 
     * @param request The request to validate
     * @throws NonRetryableException if validation fails
     */
    public static void validate(InferenceRequest request) throws NonRetryableException {
        if (request == null) {
            throw new NonRetryableException("VALIDATION_ERROR", "InferenceRequest cannot be null");
        }
        
        if (request.getModel() == null || request.getModel().trim().isEmpty()) {
            throw new NonRetryableException("VALIDATION_ERROR", "Model name is required");
        }
        
        if (request.getMessages() == null || request.getMessages().isEmpty()) {
            throw new NonRetryableException("VALIDATION_ERROR", "At least one message is required");
        }
        
                // Validate temperature range
                if (request.getParameters().containsKey("temperature")) {
                    Double temp = (Double) request.getParameters().get("temperature");
                    if (temp != null && (temp < 0.0 || temp > 2.0)) {
                        throw new NonRetryableException("VALIDATION_ERROR",
                            "Temperature must be between 0.0 and 2.0, got: " + temp);
                    }
                }        
                // Validate max tokens
                if (request.getParameters().containsKey("max_tokens")) {
                    Integer maxTokens = (Integer) request.getParameters().get("max_tokens");
                    if (maxTokens != null && maxTokens <= 0) {
                        throw new NonRetryableException("VALIDATION_ERROR",
                            "Max tokens must be positive, got: " + maxTokens);
                    }
                }    }
}
