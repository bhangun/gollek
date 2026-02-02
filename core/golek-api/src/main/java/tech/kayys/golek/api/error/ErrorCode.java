package tech.kayys.golek.api.error;

/**
 * Comprehensive error code enumeration for all failure scenarios.
 * 
 * <p>
 * Error codes follow the pattern: CATEGORY_NNN
 * <ul>
 * <li>MODEL_xxx: Model-related errors</li>
 * <li>TENSOR_xxx: Tensor operation errors</li>
 * <li>DEVICE_xxx: Hardware/device errors</li>
 * <li>QUOTA_xxx: Resource quota errors</li>
 * <li>AUTH_xxx: Authentication/authorization errors</li>
 * <li>INIT_xxx: Initialization errors</li>
 * <li>RUNTIME_xxx: Runtime execution errors</li>
 * <li>STORAGE_xxx: Storage/persistence errors</li>
 * <li>CONVERSION_xxx: Model conversion errors</li>
 * </ul>
 * 
 * @author bhangun
 * @since 1.0.0
 */
public enum ErrorCode {

    // ===== Model Errors (404, 400) =====
    MODEL_NOT_FOUND(404, "MODEL_001", "Model not found", false),
    MODEL_VERSION_NOT_FOUND(404, "MODEL_002", "Model version not found", false),
    MODEL_INVALID_FORMAT(400, "MODEL_003", "Invalid model format", false),
    MODEL_CORRUPTED(400, "MODEL_004", "Model file corrupted", false),
    MODEL_TOO_LARGE(400, "MODEL_005", "Model exceeds size limit", false),
    MODEL_SIGNATURE_INVALID(403, "MODEL_006", "Model signature verification failed", false),
    MODEL_NOT_COMPATIBLE(400, "MODEL_007", "Model not compatible with selected runner", false),

    // ===== Tensor Errors (400) =====
    TENSOR_SHAPE_MISMATCH(400, "TENSOR_001", "Tensor shape mismatch", false),
    TENSOR_TYPE_MISMATCH(400, "TENSOR_002", "Tensor data type mismatch", false),
    TENSOR_INVALID_DATA(400, "TENSOR_003", "Invalid tensor data", false),
    TENSOR_SIZE_MISMATCH(400, "TENSOR_004", "Tensor size does not match shape", false),
    TENSOR_CONVERSION_FAILED(500, "TENSOR_005", "Tensor conversion failed", true),
    TENSOR_MISSING_INPUT(400, "TENSOR_006", "Required input tensor missing", false),

    // ===== Device Errors (503, 500) =====
    DEVICE_NOT_AVAILABLE(503, "DEVICE_001", "Requested device not available", true),
    DEVICE_OUT_OF_MEMORY(503, "DEVICE_002", "Device out of memory", true),
    DEVICE_INITIALIZATION_FAILED(500, "DEVICE_003", "Device initialization failed", false),
    DEVICE_DRIVER_ERROR(500, "DEVICE_004", "Device driver error", true),
    GPU_NOT_FOUND(503, "DEVICE_005", "GPU not found", false),
    TPU_NOT_AVAILABLE(503, "DEVICE_006", "TPU not available", false),
    NPU_NOT_SUPPORTED(501, "DEVICE_007", "NPU not supported on this platform", false),

    // ===== Quota Errors (429) =====
    QUOTA_EXCEEDED(429, "QUOTA_001", "Quota exceeded", false),
    RATE_LIMIT_EXCEEDED(429, "QUOTA_002", "Rate limit exceeded", false),
    CONCURRENT_REQUESTS_EXCEEDED(429, "QUOTA_003", "Too many concurrent requests", true),
    STORAGE_QUOTA_EXCEEDED(429, "QUOTA_004", "Storage quota exceeded", false),
    COMPUTE_QUOTA_EXCEEDED(429, "QUOTA_005", "Compute quota exceeded", true),

    // ===== Authentication & Authorization (401, 403) =====
    AUTH_TOKEN_INVALID(401, "AUTH_001", "Invalid authentication token", false),
    AUTH_TOKEN_EXPIRED(401, "AUTH_002", "Authentication token expired", false),
    AUTH_TENANT_NOT_FOUND(401, "AUTH_003", "Tenant not found", false),
    AUTH_PERMISSION_DENIED(403, "AUTH_004", "Permission denied", false),
    AUTH_TENANT_SUSPENDED(403, "AUTH_005", "Tenant account suspended", false),

    // ===== Initialization Errors (500) =====
    INIT_RUNNER_FAILED(500, "INIT_001", "Runner initialization failed", false),
    INIT_MODEL_LOAD_FAILED(500, "INIT_002", "Model loading failed", true),
    INIT_NATIVE_LIBRARY_FAILED(500, "INIT_003", "Native library loading failed", false),
    INIT_CONFIGURATION_INVALID(500, "INIT_004", "Invalid configuration", false),
    INIT_DEPENDENCY_MISSING(500, "INIT_005", "Required dependency missing", false),

    // ===== Runtime Execution Errors (500, 504) =====
    RUNTIME_INFERENCE_FAILED(500, "RUNTIME_001", "Inference execution failed", true),
    RUNTIME_TIMEOUT(504, "RUNTIME_002", "Inference request timeout", true),
    RUNTIME_OUT_OF_MEMORY(500, "RUNTIME_003", "Out of memory during inference", true),
    RUNTIME_NATIVE_CRASH(500, "RUNTIME_004", "Native library crashed", true),
    RUNTIME_INVALID_STATE(500, "RUNTIME_005", "Invalid runner state", false),
    RUNTIME_BATCH_SIZE_EXCEEDED(400, "RUNTIME_006", "Batch size exceeds limit", false),

    // ===== Storage Errors (500, 503) =====
    STORAGE_READ_FAILED(500, "STORAGE_001", "Failed to read from storage", true),
    STORAGE_WRITE_FAILED(500, "STORAGE_002", "Failed to write to storage", true),
    STORAGE_NOT_FOUND(404, "STORAGE_003", "Storage resource not found", false),
    STORAGE_CONNECTION_FAILED(503, "STORAGE_004", "Storage connection failed", true),
    STORAGE_PERMISSION_DENIED(403, "STORAGE_005", "Storage permission denied", false),

    // ===== Model Conversion Errors (500) =====
    CONVERSION_FAILED(500, "CONVERSION_001", "Model conversion failed", true),
    CONVERSION_FORMAT_NOT_SUPPORTED(400, "CONVERSION_002", "Target format not supported", false),
    CONVERSION_TIMEOUT(504, "CONVERSION_003", "Model conversion timeout", true),
    CONVERSION_VALIDATION_FAILED(500, "CONVERSION_004", "Converted model validation failed", false),
    QUANTIZATION_FAILED(500, "CONVERSION_005", "Model quantization failed", true),

    // ===== Request Validation Errors (400) =====
    VALIDATION_MISSING_FIELD(400, "VALIDATION_001", "Required field missing", false),
    VALIDATION_INVALID_FORMAT(400, "VALIDATION_002", "Invalid field format", false),
    VALIDATION_CONSTRAINT_VIOLATION(400, "VALIDATION_003", "Validation constraint violated", false),

    // ===== Circuit Breaker & Resilience (503) =====
    CIRCUIT_BREAKER_OPEN(503, "CIRCUIT_001", "Circuit breaker open", true),
    ALL_RUNNERS_FAILED(503, "CIRCUIT_002", "All runner attempts failed", true),
    FALLBACK_FAILED(503, "CIRCUIT_003", "Fallback execution failed", true),

    // ===== Internal Errors (500) =====
    INTERNAL_ERROR(500, "INTERNAL_001", "Internal server error", true),
    DATABASE_ERROR(500, "INTERNAL_002", "Database error", true),
    CACHE_ERROR(500, "INTERNAL_003", "Cache error", true),
    SERIALIZATION_ERROR(500, "INTERNAL_004", "Serialization error", true);

    private final int httpStatus;
    private final String code;
    private final String defaultMessage;
    private final boolean retryable;

    ErrorCode(int httpStatus, String code, String defaultMessage, boolean retryable) {
        this.httpStatus = httpStatus;
        this.code = code;
        this.defaultMessage = defaultMessage;
        this.retryable = retryable;
    }

    /**
     * Get ErrorCode from code string.
     */
    public static ErrorCode fromCode(String code) {
        for (ErrorCode errorCode : values()) {
            if (errorCode.code.equals(code)) {
                return errorCode;
            }
        }
        return INTERNAL_ERROR;
    }

    public boolean isClientError() {
        return httpStatus >= 400 && httpStatus < 500;
    }

    public boolean isServerError() {
        return httpStatus >= 500 && httpStatus < 600;
    }

    public int getHttpStatus() {
        return httpStatus;
    }

    public String getCode() {
        return code;
    }

    public String getDefaultMessage() {
        return defaultMessage;
    }

    public boolean isRetryable() {
        return retryable;
    }
}
