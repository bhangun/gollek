package tech.kayys.wayang.tool.entity;

import java.util.Set;

import jakarta.persistence.Column;
import jakarta.persistence.ElementCollection;
import jakarta.persistence.Embeddable;
import jakarta.persistence.JoinTable;
import jakarta.persistence.Table;

@Embeddable
@Table(name = "mcp_tool_guardrails") // This is just for reference, embeddables don't create separate tables
public class ToolGuardrails {
    @Column(name = "rate_limit_per_minute")
    private Integer rateLimitPerMinute = 100;

    @Column(name = "rate_limit_per_hour")
    private Integer rateLimitPerHour = 1000;

    @Column(name = "max_request_size_kb")
    private Integer maxRequestSizeKb = 1024; // 1MB

    @Column(name = "max_response_size_kb")
    private Integer maxResponseSizeKb = 10240; // 10MB

    @Column(name = "timeout_ms")
    private Integer timeoutMs = 30000; // 30 seconds

    @ElementCollection
    @JoinTable(name = "mcp_tool_allowed_domains")
    @Column(name = "domain")
    private Set<String> allowedDomains;

    @ElementCollection
    @JoinTable(name = "mcp_tool_pii_patterns")
    @Column(name = "pattern")
    private Set<String> piiPatterns;

    @Column(name = "require_approval")
    private Boolean requireApproval = false;

    @Column(name = "validate_input_schema")
    private Boolean validateInputSchema = true;

    @Column(name = "validate_output_schema")
    private Boolean validateOutputSchema = true;

    @Column(name = "sanitize_input")
    private Boolean sanitizeInput = true;

    @Column(name = "allow_redirects")
    private Boolean allowRedirects = false;

    @Column(name = "log_requests")
    private Boolean logRequests = true;

    @Column(name = "max_retries")
    private Integer maxRetries = 3;

    @Column(name = "redact_pii")
    private Boolean redactPii = false;

    @Column(name = "enable_audit_logging")
    private Boolean enableAuditLogging = true;

    // Getters and setters
    public Boolean isValidateInputSchema() {
        return validateInputSchema;
    }

    public void setValidateInputSchema(Boolean validateInputSchema) {
        this.validateInputSchema = validateInputSchema;
    }

    public Boolean isValidateOutputSchema() {
        return validateOutputSchema;
    }

    public void setValidateOutputSchema(Boolean validateOutputSchema) {
        this.validateOutputSchema = validateOutputSchema;
    }

    public Boolean isSanitizeInput() {
        return sanitizeInput;
    }

    public void setSanitizeInput(Boolean sanitizeInput) {
        this.sanitizeInput = sanitizeInput;
    }

    public Boolean isAllowRedirects() {
        return allowRedirects;
    }

    public void setAllowRedirects(Boolean allowRedirects) {
        this.allowRedirects = allowRedirects;
    }

    public Boolean isLogRequests() {
        return logRequests;
    }

    public void setLogRequests(Boolean logRequests) {
        this.logRequests = logRequests;
    }

    public Integer getMaxRetries() {
        return maxRetries;
    }

    public void setMaxRetries(Integer maxRetries) {
        this.maxRetries = maxRetries;
    }

    public Boolean isRedactPii() {
        return redactPii;
    }

    public void setRedactPii(Boolean redactPii) {
        this.redactPii = redactPii;
    }

    public Integer getMaxExecutionTimeMs() {
        return timeoutMs;
    }

    public void setMaxExecutionTimeMs(Integer maxExecutionTimeMs) {
        this.timeoutMs = maxExecutionTimeMs;
    }

    public long getMaxResponseSizeBytes() {
        return (long) maxResponseSizeKb * 1024;
    }

    public Integer getRateLimitPerMinute() {
        return rateLimitPerMinute;
    }

    public void setRateLimitPerMinute(Integer rateLimitPerMinute) {
        this.rateLimitPerMinute = rateLimitPerMinute;
    }

    public Integer getRateLimitPerHour() {
        return rateLimitPerHour;
    }

    public void setRateLimitPerHour(Integer rateLimitPerHour) {
        this.rateLimitPerHour = rateLimitPerHour;
    }

    public Integer getMaxRequestSizeKb() {
        return maxRequestSizeKb;
    }

    public void setMaxRequestSizeKb(Integer maxRequestSizeKb) {
        this.maxRequestSizeKb = maxRequestSizeKb;
    }

    public Integer getMaxResponseSizeKb() {
        return maxResponseSizeKb;
    }

    public void setMaxResponseSizeKb(Integer maxResponseSizeKb) {
        this.maxResponseSizeKb = maxResponseSizeKb;
    }

    public Integer getTimeoutMs() {
        return timeoutMs;
    }

    public void setTimeoutMs(Integer timeoutMs) {
        this.timeoutMs = timeoutMs;
    }

    public Set<String> getAllowedDomains() {
        return allowedDomains;
    }

    public void setAllowedDomains(Set<String> allowedDomains) {
        this.allowedDomains = allowedDomains;
    }

    public Set<String> getPiiPatterns() {
        return piiPatterns;
    }

    public void setPiiPatterns(Set<String> piiPatterns) {
        this.piiPatterns = piiPatterns;
    }

    public Boolean getRequireApproval() {
        return requireApproval;
    }

    public void setRequireApproval(Boolean requireApproval) {
        this.requireApproval = requireApproval;
    }

    public Boolean getEnableAuditLogging() {
        return enableAuditLogging;
    }

    public void setEnableAuditLogging(Boolean enableAuditLogging) {
        this.enableAuditLogging = enableAuditLogging;
    }
}
