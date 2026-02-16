package tech.kayys.wayang.tool.entity;

import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Table;
import jakarta.validation.constraints.NotNull;
import tech.kayys.wayang.error.ErrorCode;
import tech.kayys.wayang.tool.dto.InvocationStatus;

/**
 * Tool invocation record for audit and billing
 */
@Entity
@Table(name = "mcp_tool_invocations", indexes = {
        @Index(name = "idx_invocations_tenant", columnList = "tenant_id"),
        @Index(name = "idx_invocations_tool", columnList = "tool_id"),
        @Index(name = "idx_invocations_timestamp", columnList = "invoked_at")
})
public class ToolInvocation extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(name = "invocation_id")
    private UUID invocationId;

    @NotNull
    @Column(name = "tenant_id")
    private String requestId;

    @NotNull
    @Column(name = "tool_id")
    private String toolId;

    // Invocation context
    @Column(name = "workflow_run_id")
    private String workflowRunId; // If called from workflow

    @Column(name = "agent_id")
    private String agentId; // If called by agent

    @Column(name = "user_id")
    private String userId;

    // Request data
    @Column(name = "arguments", columnDefinition = "jsonb")
    private Map<String, Object> arguments;

    @Column(name = "request_size_bytes")
    private Integer requestSizeBytes;

    // Response data
    @Column(name = "result", columnDefinition = "jsonb")
    private Map<String, Object> result;

    @Column(name = "response_size_bytes")
    private Integer responseSizeBytes;

    // Execution details
    @Enumerated(EnumType.STRING)
    @Column(name = "status")
    private InvocationStatus status;

    @Column(name = "error_message")
    private String errorMessage;

    @Column(name = "error_code")
    private String errorCode;

    @Column(name = "http_status_code")
    private Integer httpStatusCode;

    // Performance metrics
    @Column(name = "execution_time_ms")
    private Long executionTimeMs;

    @Column(name = "network_time_ms")
    private Long networkTimeMs;

    // Timestamps
    @Column(name = "invoked_at")
    private Instant invokedAt;

    @Column(name = "completed_at")
    private Instant completedAt;

    // Billing
    @Column(name = "billable_units")
    private Integer billableUnits = 1;

    @Column(name = "cost_cents")
    private Long costCents;

    // Getters and Setters
    public UUID getInvocationId() {
        return invocationId;
    }

    public void setInvocationId(UUID invocationId) {
        this.invocationId = invocationId;
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public String getToolId() {
        return toolId;
    }

    public void setToolId(String toolId) {
        this.toolId = toolId;
    }

    public String getWorkflowRunId() {
        return workflowRunId;
    }

    public void setWorkflowRunId(String workflowRunId) {
        this.workflowRunId = workflowRunId;
    }

    public String getAgentId() {
        return agentId;
    }

    public void setAgentId(String agentId) {
        this.agentId = agentId;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public Map<String, Object> getArguments() {
        return arguments;
    }

    public void setArguments(Map<String, Object> arguments) {
        this.arguments = arguments;
    }

    public Integer getRequestSizeBytes() {
        return requestSizeBytes;
    }

    public void setRequestSizeBytes(Integer requestSizeBytes) {
        this.requestSizeBytes = requestSizeBytes;
    }

    public Map<String, Object> getResult() {
        return result;
    }

    public void setResult(Map<String, Object> result) {
        this.result = result;
    }

    public Integer getResponseSizeBytes() {
        return responseSizeBytes;
    }

    public void setResponseSizeBytes(Integer responseSizeBytes) {
        this.responseSizeBytes = responseSizeBytes;
    }

    public InvocationStatus getStatus() {
        return status;
    }

    public void setStatus(InvocationStatus status) {
        this.status = status;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public String getErrorCode() {
        return errorCode;
    }

    public void setErrorCode(String errorCode) {
        this.errorCode = errorCode;
    }

    public void setErrorCode(ErrorCode errorCode) {
        this.errorCode = errorCode != null ? errorCode.getCode() : null;
    }

    public Integer getHttpStatusCode() {
        return httpStatusCode;
    }

    public void setHttpStatusCode(Integer httpStatusCode) {
        this.httpStatusCode = httpStatusCode;
    }

    public Long getExecutionTimeMs() {
        return executionTimeMs;
    }

    public void setExecutionTimeMs(Long executionTimeMs) {
        this.executionTimeMs = executionTimeMs;
    }

    public Long getNetworkTimeMs() {
        return networkTimeMs;
    }

    public void setNetworkTimeMs(Long networkTimeMs) {
        this.networkTimeMs = networkTimeMs;
    }

    public Instant getInvokedAt() {
        return invokedAt;
    }

    public void setInvokedAt(Instant invokedAt) {
        this.invokedAt = invokedAt;
    }

    public Instant getCompletedAt() {
        return completedAt;
    }

    public void setCompletedAt(Instant completedAt) {
        this.completedAt = completedAt;
    }

    public Integer getBillableUnits() {
        return billableUnits;
    }

    public void setBillableUnits(Integer billableUnits) {
        this.billableUnits = billableUnits;
    }

    public Long getCostCents() {
        return costCents;
    }

    public void setCostCents(Long costCents) {
        this.costCents = costCents;
    }
}
