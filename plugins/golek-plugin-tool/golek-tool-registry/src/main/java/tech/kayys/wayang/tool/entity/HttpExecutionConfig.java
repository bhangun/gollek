package tech.kayys.wayang.tool.entity;

import java.util.List;
import java.util.Map;

// Removed ParameterMapping import as it is in dto
// import tech.kayys.wayang.tool.dto.ParameterMapping; // This relies on DTO being in classpath. DTO is in execution module.

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import tech.kayys.wayang.tool.dto.ParameterMapping;
import tech.kayys.wayang.tool.dto.HttpMethod;

/**
 * HTTP execution configuration embedded in MCP Tool
 */
@Embeddable
public class HttpExecutionConfig {

    @Enumerated(EnumType.STRING)
    @Column(name = "http_method")
    private HttpMethod method;

    @Column(name = "base_url")
    private String baseUrl;

    @Column(name = "path")
    private String path;

    // Template parameters (path, query, header)
    @Column(name = "parameters", columnDefinition = "jsonb")
    private List<ParameterMapping> parameters;

    // Static headers
    @Column(name = "headers", columnDefinition = "jsonb")
    private Map<String, String> headers;

    @Column(name = "content_type")
    private String contentType = "application/json";

    @Column(name = "accept")
    private String accept = "application/json";

    // Timeout configuration
    @Column(name = "timeout_ms")
    private Integer timeoutMs = 30000;

    // Retry configuration (stored as JSON)
    @Column(name = "retry_config", columnDefinition = "jsonb")
    private String retryConfig; // Store as JSON string for simplicity

    // Getters and Setters
    public HttpMethod getMethod() {
        return method;
    }

    public void setMethod(HttpMethod method) {
        this.method = method;
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public List<ParameterMapping> getParameters() {
        return parameters;
    }

    public void setParameters(List<ParameterMapping> parameters) {
        this.parameters = parameters;
    }

    public Map<String, String> getHeaders() {
        return headers;
    }

    public void setHeaders(Map<String, String> headers) {
        this.headers = headers;
    }

    public String getContentType() {
        return contentType;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public String getAccept() {
        return accept;
    }

    public void setAccept(String accept) {
        this.accept = accept;
    }

    public Integer getTimeoutMs() {
        return timeoutMs;
    }

    public void setTimeoutMs(Integer timeoutMs) {
        this.timeoutMs = timeoutMs;
    }

    public String getRetryConfig() {
        return retryConfig;
    }

    public void setRetryConfig(String retryConfig) {
        this.retryConfig = retryConfig;
    }
}
