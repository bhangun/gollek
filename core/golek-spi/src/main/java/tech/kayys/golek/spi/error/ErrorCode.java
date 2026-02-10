package tech.kayys.golek.spi.error;

import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Central registry for all Golek error codes.
 *
 * <p>
 * Pattern: CATEGORY_NNN (example: MODEL_001)
 * <ul>
 * <li>MODEL: Model-related errors</li>
 * <li>TENSOR: Tensor operation errors</li>
 * <li>DEVICE: Hardware/device errors</li>
 * <li>QUOTA: Resource quota errors</li>
 * <li>AUTH: Authentication/authorization errors</li>
 * <li>INIT: Initialization errors</li>
 * <li>RUNTIME: Runtime execution errors</li>
 * <li>STORAGE: Storage/persistence errors</li>
 * <li>CONVERSION: Model conversion errors</li>
 * <li>VALIDATION: Request validation errors</li>
 * <li>CIRCUIT: Circuit breaker and resilience errors</li>
 * <li>PROVIDER: Provider integration errors</li>
 * <li>ROUTING: Routing and registry errors</li>
 * <li>PLUGIN: Plugin lifecycle errors</li>
 * <li>CONFIG: Configuration errors</li>
 * <li>NETWORK: Network and transport errors</li>
 * <li>STREAM: Streaming errors</li>
 * <li>INTERNAL: Internal/server errors</li>
 * </ul>
 *
 * <p>
 * Rules:
 * <ul>
 * <li>Codes are unique.</li>
 * <li>Code prefix must match its category.</li>
 * </ul>
 *
 * @author Bhangun
 * @since 1.0.0
 */
public enum ErrorCode {

    // ===== Model Errors (404, 400) =====
    MODEL_NOT_FOUND(ErrorCategory.MODEL, 404, "MODEL_001", "Model not found", false),
    MODEL_VERSION_NOT_FOUND(ErrorCategory.MODEL, 404, "MODEL_002", "Model version not found", false),
    MODEL_INVALID_FORMAT(ErrorCategory.MODEL, 400, "MODEL_003", "Invalid model format", false),
    MODEL_CORRUPTED(ErrorCategory.MODEL, 400, "MODEL_004", "Model file corrupted", false),
    MODEL_TOO_LARGE(ErrorCategory.MODEL, 400, "MODEL_005", "Model exceeds size limit", false),
    MODEL_SIGNATURE_INVALID(ErrorCategory.MODEL, 403, "MODEL_006", "Model signature verification failed", false),
    MODEL_NOT_COMPATIBLE(ErrorCategory.MODEL, 400, "MODEL_007", "Model not compatible with selected runner", false),

    // ===== Tensor Errors (400) =====
    TENSOR_SHAPE_MISMATCH(ErrorCategory.TENSOR, 400, "TENSOR_001", "Tensor shape mismatch", false),
    TENSOR_TYPE_MISMATCH(ErrorCategory.TENSOR, 400, "TENSOR_002", "Tensor data type mismatch", false),
    TENSOR_INVALID_DATA(ErrorCategory.TENSOR, 400, "TENSOR_003", "Invalid tensor data", false),
    TENSOR_SIZE_MISMATCH(ErrorCategory.TENSOR, 400, "TENSOR_004", "Tensor size does not match shape", false),
    TENSOR_CONVERSION_FAILED(ErrorCategory.TENSOR, 500, "TENSOR_005", "Tensor conversion failed", true),
    TENSOR_MISSING_INPUT(ErrorCategory.TENSOR, 400, "TENSOR_006", "Required input tensor missing", false),

    // ===== Device Errors (503, 500) =====
    DEVICE_NOT_AVAILABLE(ErrorCategory.DEVICE, 503, "DEVICE_001", "Requested device not available", true),
    DEVICE_OUT_OF_MEMORY(ErrorCategory.DEVICE, 503, "DEVICE_002", "Device out of memory", true),
    DEVICE_INITIALIZATION_FAILED(ErrorCategory.DEVICE, 500, "DEVICE_003", "Device initialization failed", false),
    DEVICE_DRIVER_ERROR(ErrorCategory.DEVICE, 500, "DEVICE_004", "Device driver error", true),
    GPU_NOT_FOUND(ErrorCategory.DEVICE, 503, "DEVICE_005", "GPU not found", false),
    TPU_NOT_AVAILABLE(ErrorCategory.DEVICE, 503, "DEVICE_006", "TPU not available", false),
    NPU_NOT_SUPPORTED(ErrorCategory.DEVICE, 501, "DEVICE_007", "NPU not supported on this platform", false),

    // ===== Quota Errors (429) =====
    QUOTA_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_001", "Quota exceeded", false),
    RATE_LIMIT_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_002", "Rate limit exceeded", false),
    CONCURRENT_REQUESTS_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_003", "Too many concurrent requests", true),
    STORAGE_QUOTA_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_004", "Storage quota exceeded", false),
    COMPUTE_QUOTA_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_005", "Compute quota exceeded", true),

    // ===== Authentication & Authorization (401, 403) =====
    AUTH_TOKEN_INVALID(ErrorCategory.AUTH, 401, "AUTH_001", "Invalid authentication token", false),
    AUTH_TOKEN_EXPIRED(ErrorCategory.AUTH, 401, "AUTH_002", "Authentication token expired", false),
    AUTH_TENANT_NOT_FOUND(ErrorCategory.AUTH, 401, "AUTH_003", "Tenant not found", false),
    AUTH_PERMISSION_DENIED(ErrorCategory.AUTH, 403, "AUTH_004", "Permission denied", false),
    AUTH_TENANT_SUSPENDED(ErrorCategory.AUTH, 403, "AUTH_005", "Tenant account suspended", false),

    // ===== Initialization Errors (500) =====
    INIT_RUNNER_FAILED(ErrorCategory.INIT, 500, "INIT_001", "Runner initialization failed", false),
    INIT_MODEL_LOAD_FAILED(ErrorCategory.INIT, 500, "INIT_002", "Model loading failed", true),
    INIT_NATIVE_LIBRARY_FAILED(ErrorCategory.INIT, 500, "INIT_003", "Native library loading failed", false),
    INIT_CONFIGURATION_INVALID(ErrorCategory.INIT, 500, "INIT_004", "Invalid configuration", false),
    INIT_DEPENDENCY_MISSING(ErrorCategory.INIT, 500, "INIT_005", "Required dependency missing", false),

    // ===== Runtime Execution Errors (500, 504) =====
    RUNTIME_INFERENCE_FAILED(ErrorCategory.RUNTIME, 500, "RUNTIME_001", "Inference execution failed", true),
    RUNTIME_TIMEOUT(ErrorCategory.RUNTIME, 504, "RUNTIME_002", "Inference request timeout", true),
    RUNTIME_OUT_OF_MEMORY(ErrorCategory.RUNTIME, 500, "RUNTIME_003", "Out of memory during inference", true),
    RUNTIME_NATIVE_CRASH(ErrorCategory.RUNTIME, 500, "RUNTIME_004", "Native library crashed", true),
    RUNTIME_INVALID_STATE(ErrorCategory.RUNTIME, 500, "RUNTIME_005", "Invalid runner state", false),
    RUNTIME_BATCH_SIZE_EXCEEDED(ErrorCategory.RUNTIME, 400, "RUNTIME_006", "Batch size exceeds limit", false),

    // ===== Storage Errors (500, 503) =====
    STORAGE_READ_FAILED(ErrorCategory.STORAGE, 500, "STORAGE_001", "Failed to read from storage", true),
    STORAGE_WRITE_FAILED(ErrorCategory.STORAGE, 500, "STORAGE_002", "Failed to write to storage", true),
    STORAGE_NOT_FOUND(ErrorCategory.STORAGE, 404, "STORAGE_003", "Storage resource not found", false),
    STORAGE_CONNECTION_FAILED(ErrorCategory.STORAGE, 503, "STORAGE_004", "Storage connection failed", true),
    STORAGE_PERMISSION_DENIED(ErrorCategory.STORAGE, 403, "STORAGE_005", "Storage permission denied", false),

    // ===== Model Conversion Errors (500) =====
    CONVERSION_FAILED(ErrorCategory.CONVERSION, 500, "CONVERSION_001", "Model conversion failed", true),
    CONVERSION_FORMAT_NOT_SUPPORTED(ErrorCategory.CONVERSION, 400, "CONVERSION_002", "Target format not supported",
            false),
    CONVERSION_TIMEOUT(ErrorCategory.CONVERSION, 504, "CONVERSION_003", "Model conversion timeout", true),
    CONVERSION_VALIDATION_FAILED(ErrorCategory.CONVERSION, 500, "CONVERSION_004", "Converted model validation failed",
            false),
    QUANTIZATION_FAILED(ErrorCategory.CONVERSION, 500, "CONVERSION_005", "Model quantization failed", true),

    // ===== Request Validation Errors (400) =====
    VALIDATION_MISSING_FIELD(ErrorCategory.VALIDATION, 400, "VALIDATION_001", "Required field missing", false),
    VALIDATION_INVALID_FORMAT(ErrorCategory.VALIDATION, 400, "VALIDATION_002", "Invalid field format", false),
    VALIDATION_CONSTRAINT_VIOLATION(ErrorCategory.VALIDATION, 400, "VALIDATION_003", "Validation constraint violated",
            false),

    // ===== Circuit Breaker & Resilience (503) =====
    CIRCUIT_BREAKER_OPEN(ErrorCategory.CIRCUIT, 503, "CIRCUIT_001", "Circuit breaker open", true),
    ALL_RUNNERS_FAILED(ErrorCategory.CIRCUIT, 503, "CIRCUIT_002", "All runner attempts failed", true),
    FALLBACK_FAILED(ErrorCategory.CIRCUIT, 503, "CIRCUIT_003", "Fallback execution failed", true),

    // ===== Provider Errors (400, 429, 502, 503, 504) =====
    PROVIDER_NOT_INITIALIZED(ErrorCategory.PROVIDER, 503, "PROVIDER_001", "Provider not initialized", true),
    PROVIDER_UNAVAILABLE(ErrorCategory.PROVIDER, 503, "PROVIDER_002", "Provider unavailable", true),
    PROVIDER_TIMEOUT(ErrorCategory.PROVIDER, 504, "PROVIDER_003", "Provider timeout", true),
    PROVIDER_AUTH_FAILED(ErrorCategory.PROVIDER, 401, "PROVIDER_004", "Provider authentication failed", false),
    PROVIDER_RATE_LIMITED(ErrorCategory.PROVIDER, 429, "PROVIDER_005", "Provider rate limit exceeded", true),
    PROVIDER_QUOTA_EXCEEDED(ErrorCategory.PROVIDER, 429, "PROVIDER_006", "Provider quota exceeded", false),
    PROVIDER_BAD_RESPONSE(ErrorCategory.PROVIDER, 502, "PROVIDER_007", "Provider returned invalid response", true),
    PROVIDER_STREAM_FAILED(ErrorCategory.PROVIDER, 502, "PROVIDER_008", "Provider stream failed", true),
    PROVIDER_INVALID_REQUEST(ErrorCategory.PROVIDER, 400, "PROVIDER_009", "Provider request invalid", false),
    PROVIDER_INIT_FAILED(ErrorCategory.PROVIDER, 500, "PROVIDER_010", "Provider initialization failed", false),

    // ===== Routing & Registry Errors (403, 404, 503) =====
    ROUTING_NO_COMPATIBLE_PROVIDER(ErrorCategory.ROUTING, 503, "ROUTING_001", "No compatible provider available", true),
    ROUTING_PROVIDER_NOT_FOUND(ErrorCategory.ROUTING, 404, "ROUTING_002", "Provider not found", false),
    ROUTING_POLICY_REJECTED(ErrorCategory.ROUTING, 403, "ROUTING_003", "Routing policy rejected request", false),

    // ===== Plugin Errors (500, 504) =====
    PLUGIN_INITIALIZATION_FAILED(ErrorCategory.PLUGIN, 500, "PLUGIN_001", "Plugin initialization failed", false),
    PLUGIN_EXECUTION_FAILED(ErrorCategory.PLUGIN, 500, "PLUGIN_002", "Plugin execution failed", true),
    PLUGIN_INVALID_CONFIGURATION(ErrorCategory.PLUGIN, 500, "PLUGIN_003", "Plugin configuration invalid", false),

    // ===== Configuration Errors (400, 500) =====
    CONFIG_MISSING(ErrorCategory.CONFIG, 500, "CONFIG_001", "Required configuration missing", false),
    CONFIG_INVALID(ErrorCategory.CONFIG, 500, "CONFIG_002", "Invalid configuration value", false),
    CONFIG_UNSUPPORTED(ErrorCategory.CONFIG, 400, "CONFIG_003", "Unsupported configuration", false),

    // ===== Network Errors (502, 503, 504) =====
    NETWORK_TIMEOUT(ErrorCategory.NETWORK, 504, "NETWORK_001", "Network timeout", true),
    NETWORK_UNREACHABLE(ErrorCategory.NETWORK, 503, "NETWORK_002", "Network unreachable", true),
    NETWORK_DNS_FAILED(ErrorCategory.NETWORK, 503, "NETWORK_003", "DNS resolution failed", true),
    NETWORK_TLS_FAILED(ErrorCategory.NETWORK, 502, "NETWORK_004", "TLS handshake failed", false),
    NETWORK_PROTOCOL_ERROR(ErrorCategory.NETWORK, 502, "NETWORK_005", "Network protocol error", true),
    NETWORK_BAD_RESPONSE(ErrorCategory.NETWORK, 502, "NETWORK_006", "Network response invalid", true),

    // ===== Stream Errors (500, 502, 504) =====
    STREAM_INIT_FAILED(ErrorCategory.STREAM, 500, "STREAM_001", "Stream initialization failed", false),
    STREAM_DISCONNECTED(ErrorCategory.STREAM, 502, "STREAM_002", "Stream disconnected", true),
    STREAM_PROTOCOL_ERROR(ErrorCategory.STREAM, 502, "STREAM_003", "Stream protocol error", true),
    STREAM_TIMEOUT(ErrorCategory.STREAM, 504, "STREAM_004", "Stream timeout", true),

    // ===== Internal Errors (500) =====
    INTERNAL_ERROR(ErrorCategory.INTERNAL, 500, "INTERNAL_001", "Internal server error", true),
    DATABASE_ERROR(ErrorCategory.INTERNAL, 500, "INTERNAL_002", "Database error", true),
    CACHE_ERROR(ErrorCategory.INTERNAL, 500, "INTERNAL_003", "Cache error", true),
    SERIALIZATION_ERROR(ErrorCategory.INTERNAL, 500, "INTERNAL_004", "Serialization error", true);

    public enum ErrorCategory {
        MODEL("MODEL"),
        TENSOR("TENSOR"),
        DEVICE("DEVICE"),
        QUOTA("QUOTA"),
        AUTH("AUTH"),
        INIT("INIT"),
        RUNTIME("RUNTIME"),
        STORAGE("STORAGE"),
        CONVERSION("CONVERSION"),
        VALIDATION("VALIDATION"),
        CIRCUIT("CIRCUIT"),
        PROVIDER("PROVIDER"),
        ROUTING("ROUTING"),
        PLUGIN("PLUGIN"),
        CONFIG("CONFIG"),
        NETWORK("NETWORK"),
        STREAM("STREAM"),
        INTERNAL("INTERNAL");

        private final String prefix;

        ErrorCategory(String prefix) {
            this.prefix = prefix;
        }

        public String getPrefix() {
            return prefix;
        }
    }

    private static final Map<String, ErrorCode> BY_CODE;
    private static final Map<String, ErrorCode> ALIASES;
    private static final Map<ErrorCategory, Map<String, ErrorCode>> BY_CATEGORY;

    static {
        Map<String, ErrorCode> byCode = new HashMap<>();
        Map<String, ErrorCode> aliases = new HashMap<>();
        Map<ErrorCategory, Map<String, ErrorCode>> byCategory = new EnumMap<>(ErrorCategory.class);

        for (ErrorCategory category : ErrorCategory.values()) {
            byCategory.put(category, new HashMap<>());
        }

        for (ErrorCode errorCode : values()) {
            if (!errorCode.code.startsWith(errorCode.category.getPrefix() + "_")) {
                throw new IllegalStateException(
                        "ErrorCode prefix mismatch: " + errorCode.name() + " -> " + errorCode.code);
            }
            ErrorCode previous = byCode.put(errorCode.code, errorCode);
            if (previous != null) {
                throw new IllegalStateException(
                        "Duplicate error code: " + errorCode.code + " (" + errorCode.name() + ", " + previous.name()
                                + ")");
            }
            byCategory.get(errorCode.category).put(errorCode.code, errorCode);
        }

        // Legacy or external aliases mapped to canonical codes
        aliases.put("PROVIDER_UNAVAILABLE", PROVIDER_UNAVAILABLE);
        aliases.put("PROVIDER_TIMEOUT", PROVIDER_TIMEOUT);
        aliases.put("PROVIDER_RATE_LIMIT_EXCEEDED", PROVIDER_RATE_LIMITED);
        aliases.put("PROVIDER_QUOTA_EXCEEDED", PROVIDER_QUOTA_EXCEEDED);
        aliases.put("PROVIDER_AUTH_FAILED", PROVIDER_AUTH_FAILED);
        aliases.put("PROVIDER_ERROR", PROVIDER_BAD_RESPONSE);
        aliases.put("AUTH_FAILED", PROVIDER_AUTH_FAILED);
        aliases.put("RATE_LIMIT_EXCEEDED", RATE_LIMIT_EXCEEDED);
        aliases.put("QUOTA_EXCEEDED", QUOTA_EXCEEDED);
        aliases.put("NO_COMPATIBLE_PROVIDER", ROUTING_NO_COMPATIBLE_PROVIDER);

        BY_CODE = Collections.unmodifiableMap(byCode);
        ALIASES = Collections.unmodifiableMap(aliases);

        Map<ErrorCategory, Map<String, ErrorCode>> immutableByCategory = new EnumMap<>(ErrorCategory.class);
        for (Map.Entry<ErrorCategory, Map<String, ErrorCode>> entry : byCategory.entrySet()) {
            immutableByCategory.put(entry.getKey(), Collections.unmodifiableMap(entry.getValue()));
        }
        BY_CATEGORY = Collections.unmodifiableMap(immutableByCategory);
    }

    private final ErrorCategory category;
    private final int httpStatus;
    private final String code;
    private final String defaultMessage;
    private final boolean retryable;

    ErrorCode(ErrorCategory category, int httpStatus, String code, String defaultMessage, boolean retryable) {
        this.category = Objects.requireNonNull(category, "category");
        this.httpStatus = httpStatus;
        this.code = Objects.requireNonNull(code, "code");
        this.defaultMessage = Objects.requireNonNull(defaultMessage, "defaultMessage");
        this.retryable = retryable;
    }

    /**
     * Get ErrorCode from code string.
     * Returns INTERNAL_ERROR when the code is unknown.
     */
    public static ErrorCode fromCode(String code) {
        return fromCode(code, INTERNAL_ERROR);
    }

    /**
     * Get ErrorCode from code string with a custom fallback.
     */
    public static ErrorCode fromCode(String code, ErrorCode fallback) {
        if (code == null || code.isBlank()) {
            return fallback;
        }
        String trimmed = code.trim();
        ErrorCode errorCode = BY_CODE.get(trimmed);
        if (errorCode != null) {
            return errorCode;
        }
        ErrorCode alias = ALIASES.get(trimmed.toUpperCase(Locale.ROOT));
        return alias != null ? alias : fallback;
    }

    /**
     * Returns an immutable map of codes within a category.
     */
    public static Map<String, ErrorCode> byCategory(ErrorCategory category) {
        Map<String, ErrorCode> codes = BY_CATEGORY.get(category);
        return codes == null ? Collections.emptyMap() : codes;
    }

    public ErrorCategory getCategory() {
        return category;
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
