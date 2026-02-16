package tech.kayys.wayang.tool.service;

import io.smallrye.mutiny.Uni;
import io.vertx.core.http.HttpMethod;
import io.vertx.core.json.JsonObject;
import io.vertx.mutiny.core.Vertx;
import io.vertx.mutiny.core.buffer.Buffer;
import io.vertx.mutiny.ext.web.client.HttpResponse;
import io.vertx.mutiny.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.apache.http.auth.AuthenticationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tech.kayys.wayang.tool.entity.HttpExecutionConfig;
import tech.kayys.wayang.tool.entity.McpTool;
import tech.kayys.wayang.tool.entity.ToolGuardrails;
import tech.kayys.wayang.tool.dto.HttpRequestContext;
import tech.kayys.wayang.tool.dto.ParameterLocation;
import tech.kayys.wayang.tool.dto.ParameterMapping;
import tech.kayys.wayang.tool.dto.InvocationStatus;
import tech.kayys.wayang.tool.exception.RateLimitExceededException;
import tech.kayys.wayang.tool.exception.ToolDisabledException;
import tech.kayys.wayang.tool.exception.InputTooLargeException;
import tech.kayys.wayang.tool.exception.ToolValidationException;
import tech.kayys.wayang.tool.exception.ResponseTooLargeException;
import tech.kayys.wayang.tool.dto.ToolExecutionRequest;
import tech.kayys.wayang.tool.dto.ToolExecutionResult;
import tech.kayys.wayang.tool.security.AuthInjector;
import tech.kayys.wayang.tool.security.NetworkSecurityFilter;
import tech.kayys.wayang.tool.security.PiiRedactor;
import tech.kayys.wayang.tool.security.RateLimiter;
import tech.kayys.wayang.tool.SchemaValidator;
import tech.kayys.wayang.tool.ToolInvocationRecorder;
import tech.kayys.wayang.tool.ToolMetricsCollector;
import tech.kayys.wayang.error.ErrorCode;

import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeoutException;

/**
 * ============================================================================
 * MCP TOOL RUNTIME EXECUTOR
 * ============================================================================
 * 
 * Executes MCP tools with full safety guarantees:
 * - Schema validation (input & output)
 * - Authentication injection
 * - Rate limiting
 * - Timeout enforcement
 * - Network security (domain allow-listing)
 * - PII redaction
 * - Audit logging
 * - Error handling & retries
 * 
 * Thread-safe and reactive (uses Mutiny)
 */
@ApplicationScoped
public class ToolExecutor {

    private static final Logger LOG = LoggerFactory.getLogger(ToolExecutor.class);

    @Inject
    Vertx vertx;

    @Inject
    ToolRegistry toolRegistry;

    @Inject
    SchemaValidator schemaValidator;

    @Inject
    AuthInjector authInjector;

    @Inject
    RateLimiter rateLimiter;

    @Inject
    NetworkSecurityFilter networkFilter;

    @Inject
    PiiRedactor piiRedactor;

    @Inject
    ToolInvocationRecorder invocationRecorder;

    @Inject
    ToolMetricsCollector metricsCollector;

    private WebClient webClient;

    /**
     * Initialize HTTP client
     */
    @jakarta.annotation.PostConstruct
    void init() {
        WebClientOptions options = new WebClientOptions()
                .setFollowRedirects(false) // Prevent open redirect attacks
                .setTrustAll(false) // Enforce SSL validation
                .setConnectTimeout(10000)
                .setIdleTimeout(60000)
                .setMaxPoolSize(100)
                .setUserAgent("Gamelan-MCP/1.0");

        this.webClient = WebClient.create(vertx, options);
    }

    /**
     * Execute MCP tool
     */
    public Uni<ToolExecutionResult> execute(ToolExecutionRequest request) {
        LOG.info("Executing tool: {} for tenant: {}",
                request.toolId(), request.requestId());

        Instant startTime = Instant.now();

        return toolRegistry.resolveTool(request.toolId(), request.requestId())
                .flatMap(tool ->
                // Pre-execution validation
                validateAndPrepare(tool, request)
                        .flatMap(validatedRequest ->
                        // Execute with guardrails
                        executeWithGuardrails(tool, validatedRequest, startTime)))
                .onFailure().recoverWithItem(throwable -> handleExecutionFailure(
                        request.toolId(),
                        throwable,
                        Duration.between(startTime, Instant.now())))
                .invoke(result ->
                // Record invocation for audit & billing
                recordInvocation(request, result, startTime));
    }

    /**
     * Validate request and prepare for execution
     */
    private Uni<ValidatedToolRequest> validateAndPrepare(
            McpTool tool,
            ToolExecutionRequest request) {

        return Uni.createFrom().item(() -> {
            // Check if tool is enabled
            if (!tool.isEnabled()) {
                throw new ToolDisabledException(
                        "Tool is disabled: " + tool.getToolId());
            }

            // Always validate input schema if available
            if (tool.getInputSchema() != null && !tool.getInputSchema().isEmpty()) {
                schemaValidator.validate(
                        tool.getInputSchema(),
                        request.arguments());
            }

            // Check rate limits
            rateLimiter.checkLimit(
                    request.requestId(),
                    tool.getToolId(),
                    tool.getGuardrails());

            // Sanitize input (now always performed if requested via context or logic)
            Map<String, Object> sanitizedArgs = sanitizeInput(request.arguments());

            // Check input size (KB)
            int inputSizeKb = calculateSize(sanitizedArgs) / 1024;
            if (inputSizeKb > tool.getGuardrails().getMaxRequestSizeKb()) {
                throw new InputTooLargeException(
                        "Input size " + inputSizeKb + "KB exceeds limit of "
                                + tool.getGuardrails().getMaxRequestSizeKb() + "KB");
            }

            return new ValidatedToolRequest(
                    tool.getToolId(),
                    request.requestId(),
                    request.userId(),
                    sanitizedArgs,
                    request.workflowRunId(),
                    request.agentId());
        });
    }

    /**
     * Execute tool with all guardrails
     */
    private Uni<ToolExecutionResult> executeWithGuardrails(
            McpTool tool,
            ValidatedToolRequest request,
            Instant startTime) {

        // Build HTTP request
        return buildHttpRequest(tool, request.arguments())
                .flatMap(httpRequest ->
                // Inject authentication
                authInjector.injectAuth(httpRequest, tool.getAuthProfileId()))
                .flatMap(authenticatedRequest ->
                // Apply network security
                networkFilter.validateRequest(authenticatedRequest, tool.getGuardrails())
                        .replaceWith(authenticatedRequest))
                .flatMap(secureRequest ->
                // Execute HTTP call with timeout
                executeHttpCall(secureRequest, tool.getGuardrails()))
                .flatMap(response ->
                // Validate and process response
                processResponse(tool, response, startTime));
    }

    /**
     * Build HTTP request from tool config and arguments
     */
    private Uni<HttpRequestContext> buildHttpRequest(
            McpTool tool,
            Map<String, Object> arguments) {

        return Uni.createFrom().item(() -> {
            HttpExecutionConfig config = tool.getExecutionConfig();

            // Build URL with path parameters
            String url = buildUrl(config, arguments);

            // Extract query parameters
            Map<String, String> queryParams = extractQueryParams(config, arguments);

            // Extract headers
            Map<String, String> headers = new HashMap<>(config.getHeaders());
            extractHeaderParams(config, arguments, headers);

            // Extract body
            Object body = extractBody(config, arguments);

            return new HttpRequestContext(
                    config.getMethod(),
                    url,
                    queryParams,
                    headers,
                    body,
                    config.getContentType());
        });
    }

    /**
     * Execute HTTP call with retries and timeout
     */
    private Uni<HttpResponse<Buffer>> executeHttpCall(
            HttpRequestContext request,
            ToolGuardrails guardrails) {

        return Uni.createFrom().item(() -> {
            var httpRequest = webClient
                    .requestAbs(
                            HttpMethod.valueOf(request.method().name()),
                            request.url());

            // Add query parameters
            request.queryParams().forEach(httpRequest::addQueryParam);

            // Add headers
            request.headers().forEach(httpRequest::putHeader);

            // Set timeout
            httpRequest.timeout(guardrails.getMaxExecutionTimeMs());

            return httpRequest;
        })
                .flatMap(httpRequest -> {
                    // Send request with or without body
                    if (request.body() != null) {
                        return httpRequest.sendJson(request.body());
                    } else {
                        return httpRequest.send();
                    }
                })
                .onFailure(TimeoutException.class).retry()
                .withBackOff(Duration.ofMillis(1000), Duration.ofSeconds(10))
                .atMost(guardrails.getMaxRetries());
    }

    /**
     * Process HTTP response
     */
    private Uni<ToolExecutionResult> processResponse(
            McpTool tool,
            HttpResponse<Buffer> response,
            Instant startTime) {

        return Uni.createFrom().item(() -> {
            long executionTime = Duration.between(startTime, Instant.now()).toMillis();

            // Check response size
            int responseSize = response.body() != null ? response.body().length() : 0;

            if (responseSize > tool.getGuardrails().getMaxResponseSizeBytes()) {
                throw new ResponseTooLargeException(
                        "Response size " + responseSize + " exceeds limit");
            }

            // Parse response
            Map<String, Object> output = parseResponse(response, tool);

            // Validate output schema
            if (tool.getGuardrails().isValidateOutputSchema()) {
                schemaValidator.validate(tool.getOutputSchema(), output);
            }

            // Redact PII if needed
            if (tool.getGuardrails().isRedactPii()) {
                output = piiRedactor.redact(
                        output,
                        tool.getGuardrails().getPiiPatterns());
            }

            // Build result
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("httpStatus", response.statusCode());
            metadata.put("responseSize", responseSize);
            metadata.put("headers", response.headers().names());

            if (response.statusCode() >= 200 && response.statusCode() < 300) {
                return ToolExecutionResult.success(
                        tool.getToolId(),
                        output,
                        executionTime);
            } else {
                return ToolExecutionResult.failure(
                        tool.getToolId(),
                        InvocationStatus.FAILURE,
                        "HTTP " + response.statusCode() + ": " + response.statusMessage(),
                        executionTime);
            }
        });
    }

    /**
     * Parse HTTP response based on content type
     */
    private Map<String, Object> parseResponse(
            HttpResponse<Buffer> response,
            McpTool tool) {

        if (response.body() == null || response.body().length() == 0) {
            return Map.of();
        }

        String contentType = response.getHeader("Content-Type");

        if (contentType != null && contentType.contains("application/json")) {
            try {
                JsonObject json = response.bodyAsJsonObject();
                return json != null ? json.getMap() : Map.of();
            } catch (Exception e) {
                LOG.warn("Failed to parse JSON response", e);
                return Map.of("_raw", response.bodyAsString());
            }
        } else {
            // Non-JSON response
            return Map.of(
                    "_raw", response.bodyAsString(),
                    "_contentType", contentType);
        }
    }

    /**
     * Build URL with path parameters
     */
    private String buildUrl(HttpExecutionConfig config, Map<String, Object> arguments) {
        String url = config.getBaseUrl() + config.getPath();

        // Replace path parameters
        if (config.getParameters() != null) {
            for (ParameterMapping param : config.getParameters()) {
                if (param.getLocation() == ParameterLocation.PATH) {
                    Object value = arguments.get(param.getName());
                    if (value != null) {
                        url = url.replace(
                                "{" + param.getMappedName() + "}",
                                String.valueOf(value));
                    }
                }
            }
        }

        return url;
    }

    /**
     * Extract query parameters
     */
    private Map<String, String> extractQueryParams(
            HttpExecutionConfig config,
            Map<String, Object> arguments) {

        Map<String, String> queryParams = new HashMap<>();

        if (config.getParameters() != null) {
            for (ParameterMapping param : config.getParameters()) {
                if (param.getLocation() == ParameterLocation.QUERY) {
                    Object value = arguments.get(param.getName());
                    if (value != null) {
                        queryParams.put(param.getMappedName(), String.valueOf(value));
                    }
                }
            }
        }

        return queryParams;
    }

    /**
     * Extract header parameters
     */
    private void extractHeaderParams(
            HttpExecutionConfig config,
            Map<String, Object> arguments,
            Map<String, String> headers) {

        if (config.getParameters() != null) {
            for (ParameterMapping param : config.getParameters()) {
                if (param.getLocation() == ParameterLocation.HEADER) {
                    Object value = arguments.get(param.getName());
                    if (value != null) {
                        headers.put(param.getMappedName(), String.valueOf(value));
                    }
                }
            }
        }
    }

    /**
     * Extract request body
     */
    private Object extractBody(
            HttpExecutionConfig config,
            Map<String, Object> arguments) {

        // Collect non-parameter arguments as body
        Set<String> paramNames = new HashSet<>();
        if (config.getParameters() != null) {
            config.getParameters().forEach(p -> paramNames.add(p.getName()));
        }

        Map<String, Object> body = new HashMap<>();
        arguments.forEach((key, value) -> {
            if (!paramNames.contains(key)) {
                body.put(key, value);
            }
        });

        return body.isEmpty() ? null : body;
    }

    /**
     * Sanitize input to prevent injection attacks
     */
    private Map<String, Object> sanitizeInput(Map<String, Object> input) {
        Map<String, Object> sanitized = new HashMap<>();

        input.forEach((key, value) -> {
            if (value instanceof String) {
                // Basic sanitization - in production use library like OWASP
                String str = (String) value;
                str = str.replace("<script", "")
                        .replace("javascript:", "")
                        .replace("onerror=", "");
                sanitized.put(key, str);
            } else {
                sanitized.put(key, value);
            }
        });

        return sanitized;
    }

    /**
     * Calculate size of arguments
     */
    private int calculateSize(Map<String, Object> arguments) {
        try {
            return new JsonObject(arguments).encode().getBytes().length;
        } catch (Exception e) {
            return 0;
        }
    }

    /**
     * Handle execution failure
     */
    private ToolExecutionResult handleExecutionFailure(
            String toolId,
            Throwable throwable,
            Duration executionTime) {

        LOG.error("Tool execution failed: {}", toolId, throwable);

        InvocationStatus status = determineFailureStatus(throwable);
        ErrorCode errorCode = mapErrorCode(throwable);

        return ToolExecutionResult.failure(
                toolId,
                status,
                throwable.getMessage(),
                executionTime.toMillis(),
                Map.of(
                        "errorCode", errorCode.getCode(),
                        "retryable", errorCode.isRetryable()));
    }

    /**
     * Determine failure status from exception
     */
    private InvocationStatus determineFailureStatus(Throwable throwable) {
        if (throwable instanceof TimeoutException) {
            return InvocationStatus.TIMEOUT;
        } else if (throwable instanceof RateLimitExceededException) {
            return InvocationStatus.RATE_LIMITED;
        } else if (throwable instanceof ToolValidationException) {
            return InvocationStatus.VALIDATION_ERROR;
        } else if (throwable instanceof AuthenticationException) {
            return InvocationStatus.AUTH_ERROR;
        } else {
            return InvocationStatus.FAILURE;
        }
    }

    private ErrorCode mapErrorCode(Throwable throwable) {
        if (throwable instanceof TimeoutException) {
            return ErrorCode.TIMEOUT;
        }
        if (throwable instanceof RateLimitExceededException) {
            return ErrorCode.RATE_LIMITED;
        }
        if (throwable instanceof ToolValidationException) {
            return ErrorCode.VALIDATION_FAILED;
        }
        if (throwable instanceof AuthenticationException) {
            return ErrorCode.SECURITY_UNAUTHORIZED;
        }
        if (throwable instanceof ToolDisabledException) {
            return ErrorCode.TOOL_NOT_FOUND;
        }
        if (throwable instanceof InputTooLargeException || throwable instanceof ResponseTooLargeException) {
            return ErrorCode.VALIDATION_FAILED;
        }
        return ErrorCode.TOOL_EXECUTION_FAILED;
    }

    /**
     * Record invocation for audit and billing
     */
    private void recordInvocation(
            ToolExecutionRequest request,
            ToolExecutionResult result,
            Instant startTime) {

        invocationRecorder.record(request, result, startTime)
                .subscribe().with(
                        v -> LOG.debug("Invocation recorded: {}", request.toolId()),
                        error -> LOG.error("Failed to record invocation", error));

        metricsCollector.collect(request.toolId(), result)
                .subscribe().with(
                        v -> {
                        },
                        error -> LOG.error("Failed to collect metrics", error));
    }

    /**
     * Validated tool request wrapper
     */
    private record ValidatedToolRequest(
            String toolId,
            String requestId,
            String userId,
            Map<String, Object> arguments,
            String workflowRunId,
            String agentId) {
    }
}
